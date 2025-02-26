import argparse
from pathlib import Path
import ffmpeg


def extract_frames(video_path: Path, output_dir, max_frames=30):
    """Extract frames from a video using ffmpeg-python."""
    # Create output directory for this video
    video_name = video_path.stem
    video_output_dir = Path(output_dir) / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Get video information
    probe = ffmpeg.probe(str(video_path))
    # video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    # print("video info: ", video_info)
    duration = float(probe['format']['duration'])

    # Calculate frame intervals
    num_frames = min(max_frames, int(duration))
    interval = duration / (num_frames + 1)

    # Extract frames
    for i in range(1, num_frames + 1):
        time_pos = interval * i
        output_file = video_output_dir / f"frame_{i:03d}.jpg"

        # Use ffmpeg-python's fluent interface
        (
            ffmpeg
            .input(str(video_path), ss=time_pos)
            .output(str(output_file), vframes=1, q=2, update=True)
            .overwrite_output()
            .run(quiet=True)
        )

    return video_output_dir


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--videos_dir', default='posts/videos', help='Directory containing videos')
    parser.add_argument('--output_dir', default='posts/stills', help='Directory to save extracted frames')
    parser.add_argument('--max_frames', type=int, default=5, help='Max number of frames to extract per video')
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each video
    for video_file in videos_dir.glob('*.mp4'):
        print(f"Processing {video_file}...")
        frames_dir = extract_frames(
            video_file,
            output_dir,
            args.max_frames
        )
        print(f"Extracted frames saved to {frames_dir}")

if __name__ == "__main__":
    main()