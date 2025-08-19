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
	postsQuery = `
			SELECT p.* FROM posts p
			JOIN post_tags pt ON p.id = pt.post_id
			JOIN tags t ON pt.tag_id = t.tag_id
			WHERE t.name = ?
			AND p.score > ?
			ORDER BY random()
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

func FindPostsWithTag(ctx context.Context, db *sqlx.DB, tag string, minScore int) error {
	slog.Info("finding posts with tag", "tag", tag)
	var posts []Post
	if err := db.SelectContext(ctx, &posts, postsQuery, tag, minScore); err != nil {
		return fmt.Errorf("find posts with tag %s: %w", tag, err)
	}

	for _, post := range posts[:10] {
		slog.Info("found post", "post", post)
	}

	var count []int64
	if err := db.SelectContext(ctx, &count, countQuery, tag, minScore); err != nil {
		return fmt.Errorf("count posts with tag %s: %w", tag, err)
	}
	slog.Info("count of posts with tag", "tag", tag, "count", count)
	return nil
}

type TaggedPost struct {
	Post
	Tags []any `db:"tags"`
}

func FindTagString(ctx context.Context, db *sqlx.DB, tag string, minScore int) ([]TaggedPost, error) {
	query := `
	select p.*, t.tags FROM
	(
		SELECT p.*
		FROM posts p
		JOIN post_tags pt ON p.id = pt.post_id
		JOIN tags t ON pt.tag_id = t.tag_id
		WHERE t.name = ?
		AND p.score > ?
		ORDER BY random()
		limit 5
	) p
	JOIN (
		SELECT pt.post_id, array_agg(tg.name) as tags
		FROM post_tags pt
		JOIN tags tg ON pt.tag_id = tg.tag_id
		GROUP BY post_id
	) t
	ON p.id = t.post_id
	`

	var result []TaggedPost
	if err := db.SelectContext(ctx, &result, query, tag, minScore); err != nil {
		return nil, fmt.Errorf("find tag %s: %w", tag, err)
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("tag %s not found", tag)
	}
	return result, nil
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
