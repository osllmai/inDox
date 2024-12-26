from detr import DETR
import sys
import matplotlib as plt


def main():
    # Initialize model
    detr = DETR(device="cuda" if len(sys.argv) < 2 else sys.argv[1])

    # Process image
    image_path = sys.argv[2] if len(sys.argv) > 2 else "input.jpg"
    results = detr.detect_objects(image_path)

    # Save visualization
    detr.visualize_results(results)
    plt.savefig("output.jpg")
    print("Detection completed. Results saved as output.jpg")


if __name__ == "__main__":
    main()
