package main

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"time"

	"github.com/jmoiron/sqlx"

	_ "github.com/marcboeker/go-duckdb"
)

// Database models
type Post struct {
	ID          int64     `db:"id"`
	Created_at  time.Time `db:"created_at"`
	Rating      string    `db:"rating"`
	ImageWidth  int       `db:"image_width"`
	ImageHeight int       `db:"image_height"`
	FavCount    int       `db:"fav_count"`
	FileExt     string    `db:"file_ext"`
	IsDeleted   bool      `db:"is_deleted"`
	Score       int       `db:"score"`
	UpScore     int       `db:"up_score"`
	DownScore   int       `db:"down_score"`
}

type Tag struct {
	TagID    int64  `db:"tag_id"`
	Name     string `db:"name"`
	Category string `db:"category"`
}

var Categories = []string{
	0: "general",     // most tags
	1: "artist",      // ok
	2: "contributor", // eg. modeler of an asset, not cg artist
	3: "copyright",   // eg "nintendo"
	4: "character",   // ok
	5: "species",     // ok
	6: "lore",        // rare
	7: "meta",        // ok
	8: "invalid",     // ok
}

var Ratings = map[string]string{
	"g": "general",
	"q": "questionable",
	"e": "explicit",
}

type TagCount struct {
	TagID     int64 `db:"tag_id"`
	PostCount int   `db:"post_count"`
}

type PostTag struct {
	PostID int64 `db:"post_id"`
	TagID  int64 `db:"tag_id"`
}

// queries

var (
	postsQueryHeader = `
			SELECT p.* FROM posts p
			JOIN post_tags pt ON p.id = pt.post_id
			JOIN tags t ON pt.tag_id = t.tag_id
			`
	tagNameCond = `
			WHERE t.name = ?
			`
	postsScoreCond = `
			AND p.score > ?
			`
	randCond = `
			ORDER BY random()
			`
	limitCond = `
			LIMIT ?
			`
	countQuery = `
			SELECT COUNT(*) FROM (
				select p.* from posts p
				JOIN post_tags pt ON p.id = pt.post_id
				JOIN tags t ON pt.tag_id = t.tag_id
				WHERE t.name = ?
				AND p.score > ?
			)
		`
)

type FindPostsOptions struct {
	Tags        []string
	ExcludeTags []string
	MinScore    *int
	MinFavs     *int
	MinUpScore  *int
	Limit       int
	Random      bool
	DebugQuery  bool
}

func FindPosts(ctx context.Context, db *sqlx.DB, opts FindPostsOptions) ([]Post, error) {
	queryBuilder := strings.Builder{}
	queryBuilder.WriteString(postsQueryHeader)
	args := make([]any, 0, 3)
	if len(opts.Tags) > 0 {
		tag := opts.Tags[0]
		queryBuilder.WriteString(tagNameCond)
		args = append(args, tag)
		var count []int64
		if err := db.SelectContext(ctx, &count, countQuery, tag, opts.MinScore); err != nil {
			return nil, fmt.Errorf("count posts %+v: %w", opts, err)
		}
		slog.Info("count of posts with tag", "tag", tag, "count", count)
	}
	if opts.MinScore != nil {
		queryBuilder.WriteString(postsScoreCond)
		args = append(args, *opts.MinScore)
	}
	if opts.Random {
		queryBuilder.WriteString(randCond)
	}
	limit := 10
	if opts.Limit > 0 {
		limit = opts.Limit
	}
	args = append(args, limit)
	queryBuilder.WriteString(limitCond)

	var posts []Post
	if err := db.SelectContext(ctx, &posts, queryBuilder.String(), args...); err != nil {
		return nil, fmt.Errorf("find posts %v: %w", opts, err)
	}
	return posts, nil
}

var (
	taggedPostHeader = `
		select p.*, t.tags FROM
		(
		`
	taggedPostPostHeader = `
			SELECT p.*
			FROM posts p
			JOIN post_tags pt ON p.id = pt.post_id
			JOIN tags t ON pt.tag_id = t.tag_id
			`
	taggedPostFinal = `
		) p
		JOIN (
			SELECT pt.post_id, array_agg(tg.name) as tags
			FROM post_tags pt
			JOIN tags tg ON pt.tag_id = tg.tag_id
			GROUP BY post_id
		) t
		ON p.id = t.post_id
	`
)

// DuckDB returns []any for an array_agg return type. Unmarshal to []string using a custom type.
type StringSlice []string

func (s *StringSlice) Scan(src any) error {
	switch v := src.(type) {
	case []any:
		*s = make(StringSlice, len(v))
		for i, val := range v {
			str, ok := val.(string)
			if !ok {
				return fmt.Errorf("element %d is not a string: %T", i, val)
			}
			(*s)[i] = str
		}
		return nil
	default:
		return fmt.Errorf("unsupported type for StringSlice: %T", src)
	}
}

type Int32Slice []int32

func (s *Int32Slice) Scan(src any) error {
	switch v := src.(type) {
	case []any:
		*s = make(Int32Slice, len(v))
		for i, val := range v {
			num, ok := val.(int32)
			if !ok {
				return fmt.Errorf("element %d is not an int: %T", i, val)
			}
			(*s)[i] = num
		}
		return nil
	default:
		return fmt.Errorf("unsupported type for IntSlice: %T", src)
	}
}

type TaggedPost struct {
	Post
	Tags        StringSlice `db:"tags"`
	TagCategory Int32Slice  `db:"categories"`
}

// FindPostsWithTag searches the database for posts matching the given conditions.
// Each post is returned with an array of tags in string format.
func FindPostsWithTag(ctx context.Context, db *sqlx.DB, opts FindPostsOptions) ([]TaggedPost, error) {
	if len(opts.Tags) == 0 {
		return nil, fmt.Errorf("no tags provided in options: %v", opts)
	}
	tag := opts.Tags[0]

	queryBuilder := strings.Builder{}
	args := make([]any, 0, 3)
	queryBuilder.WriteString(taggedPostHeader)
	queryBuilder.WriteString(taggedPostPostHeader)
	queryBuilder.WriteString(tagNameCond)
	args = append(args, tag)
	if opts.MinScore != nil {
		queryBuilder.WriteString(postsScoreCond)
		args = append(args, *opts.MinScore)
	}
	if opts.Random {
		queryBuilder.WriteString(randCond)
	}
	limit := 10
	if opts.Limit > 0 {
		limit = opts.Limit
	}
	args = append(args, limit)
	queryBuilder.WriteString(limitCond)
	queryBuilder.WriteString(taggedPostFinal)
	var result []TaggedPost
	q := queryBuilder.String()
	if err := db.SelectContext(ctx, &result, q, args...); err != nil {
		return nil, fmt.Errorf("find tag %s: %w", tag, err)
	}
	return result, nil
}

// FindPostsWithAllTags implements an intersection search.
func FindPostsWithAllTags(ctx context.Context, db *sqlx.DB, opts FindPostsOptions) ([]TaggedPost, error) {
	if len(opts.Tags) == 0 {
		return nil, fmt.Errorf("no tags provided in options: %v", opts)
	}

	args := []any{opts.Tags}

	// build interstitial clauses
	andDoesNotHaveTags := ""
	if len(opts.ExcludeTags) > 0 {
		andDoesNotHaveTags = `
			AND p.id NOT IN (
			SELECT pt2.post_id
			FROM post_tags pt2
			JOIN tags t2 ON pt2.tag_id = t2.tag_id
			WHERE t2.name IN (?)
		)`
		args = append(args, opts.ExcludeTags)
	}

	andScoreGreaterThan := ""
	if opts.MinScore != nil {
		andScoreGreaterThan = "AND p.score > ?"
		args = append(args, *opts.MinScore)
	}

	andFavsGreaterThan := ""
	if opts.MinFavs != nil {
		andFavsGreaterThan = "AND p.fav_count > ?"
		args = append(args, *opts.MinFavs)
	}

	args = append(args, len(opts.Tags))

	orderBy := ""
	if opts.Random {
		orderBy = "ORDER BY random()"
	}

	limit := 10
	if opts.Limit > 0 {
		limit = opts.Limit
	}
	args = append(args, limit)

	builder := strings.Builder{}
	bws := func(s string) { builder.WriteString(s) }
	bws(`
	SELECT p.*, t.tags, t.categories
	FROM (
		SELECT p.*
		FROM posts p
		JOIN post_tags pt ON p.id = pt.post_id
		JOIN tags t ON pt.tag_id = t.tag_id
		WHERE t.name IN (?)`)
	bws(andDoesNotHaveTags)
	bws(andScoreGreaterThan)
	bws(andFavsGreaterThan)
	bws(`
		GROUP BY p.id, p.created_at, p.rating, p.image_width, p.image_height,
				p.fav_count, p.file_ext, p.is_deleted, p.score, p.up_score, p.down_score
		HAVING COUNT(DISTINCT t.name) = ?
		`)
	bws(orderBy)
	bws(`
		LIMIT ?
	) p
	 JOIN (
		SELECT pt.post_id, array_agg(tg.name) as tags, array_agg(tg.category) as categories
		FROM post_tags pt
		JOIN tags tg ON pt.tag_id = tg.tag_id
		GROUP BY pt.post_id
	) t
	ON p.id = t.post_id`)

	query, args, err := sqlx.In(builder.String(), args...)
	if err != nil {
		return nil, err
	}

	var result []TaggedPost
	if opts.DebugQuery {
		fmt.Println("Query:", query)
		fmt.Println("Args:", args)
	}
	if err := db.SelectContext(ctx, &result, query, args...); err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}

	return result, nil
}

// CountPostsWithAllTags counts the total number of posts matching the search criteria
func CountPostsWithAllTags(ctx context.Context, db *sqlx.DB, opts FindPostsOptions) (int, error) {
	if len(opts.Tags) == 0 {
		return 0, fmt.Errorf("no tags provided in options: %v", opts)
	}

	args := []any{opts.Tags}

	// build interstitial clauses
	andDoesNotHaveTags := ""
	if len(opts.ExcludeTags) > 0 {
		andDoesNotHaveTags = `
			AND p.id NOT IN (
			SELECT pt2.post_id
			FROM post_tags pt2
			JOIN tags t2 ON pt2.tag_id = t2.tag_id
			WHERE t2.name IN (?)
		)`
		args = append(args, opts.ExcludeTags)
	}

	andScoreGreaterThan := ""
	if opts.MinScore != nil {
		andScoreGreaterThan = "AND p.score > ?"
		args = append(args, *opts.MinScore)
	}

	andFavsGreaterThan := ""
	if opts.MinFavs != nil {
		andFavsGreaterThan = "AND p.fav_count > ?"
		args = append(args, *opts.MinFavs)
	}

	args = append(args, len(opts.Tags))

	builder := strings.Builder{}
	bws := func(s string) { builder.WriteString(s) }
	bws(`
	SELECT COUNT(*) FROM (
		SELECT p.id
		FROM posts p
		JOIN post_tags pt ON p.id = pt.post_id
		JOIN tags t ON pt.tag_id = t.tag_id
		WHERE t.name IN (?)`)
	bws(andDoesNotHaveTags)
	bws(andScoreGreaterThan)
	bws(andFavsGreaterThan)
	bws(`
		GROUP BY p.id
		HAVING COUNT(DISTINCT t.name) = ?
	)`)

	query, args, err := sqlx.In(builder.String(), args...)
	if err != nil {
		return 0, err
	}

	var count int
	if err := db.GetContext(ctx, &count, query, args...); err != nil {
		return 0, fmt.Errorf("count query: %w", err)
	}

	return count, nil
}

type Pragma struct {
	ColumnID   int            `db:"cid"`
	Name       string         `db:"name"`
	Type       string         `db:"type"`
	NotNull    bool           `db:"notnull"`
	Default    sql.NullString `db:"dflt_value"`
	PrimaryKey bool           `db:"pk"`
}

func PrintTableInfo(ctx context.Context, db *sqlx.DB, table string) error {
	validName := regexp.MustCompile(`^[a-zA-Z0-9_]+$`)
	if !validName.MatchString(table) {
		return fmt.Errorf("invalid table name: %s", table)
	}
	stmt := fmt.Sprintf("PRAGMA table_info('%s')", table)
	var cols []Pragma
	if err := db.SelectContext(ctx, &cols, stmt); err != nil {
		return fmt.Errorf("get table info: %w", err)
	}

	b := strings.Builder{}
	for i, col := range cols {
		b.WriteString(fmt.Sprintf("%s %s", col.Name, col.Type))
		if i < len(cols)-1 {
			b.WriteString(" | ")
		}
	}
	slog.Info("table_info", "columns", b.String())
	return nil
}

func DBDump(ctx context.Context, db *sqlx.DB) error {
	if err := PrintTableInfo(ctx, db, "posts"); err != nil {
		return fmt.Errorf("print table info: %w", err)
	}

	if err := PrintTableInfo(ctx, db, "tag_counts"); err != nil {
		return fmt.Errorf("print table info: %w", err)
	}

	if err := PrintTableInfo(ctx, db, "post_tags"); err != nil {
		return fmt.Errorf("print table info: %w", err)
	}

	var tables []string
	if err := db.SelectContext(ctx, &tables, "SHOW TABLES;"); err != nil {
		return fmt.Errorf("show tables: %w", err)
	}
	slog.Info("tables in database", "tables", tables)
	return nil
}
