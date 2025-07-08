import cv2
import os
import argparse

def images_to_video(image_folder, output_path, fps=10):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    if not images:
        print("No images found.")
        return

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image_name in images:
        frame = cv2.imread(os.path.join(image_folder, image_name))
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True, help="Path to sequence images folder")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Output video file path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output video")
    args = parser.parse_args()

    images_to_video(args.image_folder, args.output_path, args.fps)