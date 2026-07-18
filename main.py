"""
main.py
-------
Entry point for the anime-face DCGAN project.

Run:
    python main.py --data_dir data --epochs 50 --batch_size 128

Your `data/` folder must contain a subfolder of images, e.g.:
    data/
        anime_faces/
            img_0001.jpg
            ...
"""

import argparse

from src import GANTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train an anime-face DCGAN")
    parser.add_argument("--data_dir", type=str, default="data/anime_faces", help="Path to folder containing image subfolder(s)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Where samples/checkpoints/models are saved")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=100, help="Size of latent noise vector")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=50, help="Save a checkpoint every N epochs")
    return parser.parse_args()


def main():
    args = parse_args()

    trainer = GANTrainer(
        data_dir=args.data_dir,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )

    trainer.train(epochs=args.epochs, save_every=args.save_every)
    trainer.save_models()


if __name__ == "__main__":
    main()