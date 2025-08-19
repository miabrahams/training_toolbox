package main

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/jmoiron/sqlx"

	"github.com/knadh/koanf/parsers/yaml"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
	_ "github.com/marcboeker/go-duckdb"
)

func load_config(path string) (*koanf.Koanf, error) {
	k := koanf.New(".")
	err := k.Load(file.Provider(path), yaml.Parser())
	return k, err
}

func main() {
	if err := run_main(); err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}
}

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

func run_main() error {
	ctx := context.Background()
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	k, err := load_config("../config.yml")
	if err != nil {
		return err
	}

	db, err := sqlx.Open("duckdb", k.String("db_path"))
	if err != nil {
		return err
	}
	defer db.Close()

	logger.Info("connected to database", "db_path", k.String("db_path"))
	if err := print_table_info(ctx, db, "tag_counts"); err != nil {
		return fmt.Errorf("print table info: %w", err)
	}

	if err := print_table_info(ctx, db, "post_tags"); err != nil {
		return fmt.Errorf("print table info: %w", err)
	}

	var tables []string
	if err := db.SelectContext(ctx, &tables, "SHOW TABLES;"); err != nil {
		return fmt.Errorf("show tables: %w", err)
	}
	logger.Info("tables in database", "tables", tables)

	var posts []Post
	if err := db.SelectContext(ctx, &posts, "SELECT * FROM (select * from posts ORDER BY RANDOM()) LIMIT 2"); err != nil {
		return err
	}

	for _, post := range posts {
		logger.Info("row", "value", post)
	}

	if err := findPostsWithTag(ctx, db, k.String("tag"), k.Int("min_score")); err != nil {
		return fmt.Errorf("find posts with tag: %w", err)
	}

	return nil
}

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

func findPostsWithTag(ctx context.Context, db *sqlx.DB, tag string, minScore int) error {
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

type pragma struct {
	ColumnID   int            `db:"cid"`
	Name       string         `db:"name"`
	Type       string         `db:"type"`
	NotNull    bool           `db:"notnull"`
	Default    sql.NullString `db:"dflt_value"`
	PrimaryKey bool           `db:"pk"`
}

func print_table_info(ctx context.Context, db *sqlx.DB, table string) error {
	slog.Info("getting table info", "table", table)
	stmt := fmt.Sprintf("PRAGMA table_info('%s')", table)
	var cols []pragma
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
