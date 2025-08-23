# ## Imports
import os
import sys
import torch
import scipy
import pickle
import shutil
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from argparse import Namespace

# Update path to include mypkg
pkg_path = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, pkg_path)

from src import *

# Load config file
config = global_config.config

def main(args=None):
    # Asserts
    if args.dataset is None or args.dataset not in ['FMNIST', 'MNIST', 'YaleFaces', 'CelebA', 'CIFAR10', 'DTD', 'EuroSAT', 'FGVCAircraft', 'Omniglot', 'PCAM', 'SEMEION', 'STL10']:
        raise Exception(f"Please specify a dataset: FMNIST, MNIST, YaleFaces, CIFAR10, DTD, EuroSAT, FGVCAircraft, Omniglot, PCAM, SEMEION, STL10, or CelebA. Not \"{args.dataset}\".")

    # Dataset options
    args.dataset_size = None # Size of dataset to use None for full dataset
    args.classes = None # [3, 8] # Select a subset of dataset classes, could be [0, 1], [3, 9], ...
    args.shape = datasets.get_dataset_shape(args.dataset)

    # Data/Plot options
    args.fft_pos_freq = False
    args.fft_normalize = True
    args.fft_centering = True

    # Integral options
    args.p_value, args.smoothing, args.radius_mult = helpers.get_radial_int_settings(args.dataset)

    args.p_value = args.custom_p_value if args.custom_p_value is not None else 2
    args.radius_mult = args.custom_radius_mult if args.custom_radius_mult is not None else 1
    args.smoothing = args.custom_smoothing if args.custom_smoothing is not None else 1

    # Change save folder
    output_dir = args.save_dir_name if args.save_dir_name is not None else args.dataset
    output_dir = str(pathlib.Path(config.top_dir) / "fft_images" / output_dir)

    config.update_top_dir(output_dir)
    config.clear_top_dir()
    config.create_output_dirs()

    print(f"Loading \"{args.dataset}\".\n")

    # Load dataset
    X_train, X_val, _ = datasets.get_dataset(dataset=args.dataset,
                                        dataset_size=args.dataset_size,
                                        classes=args.classes,
                                        val_split=args.val_split)

    # Training options
    args.batch_size = args.batch_size if args.batch_size != -1 else X_train.shape[0]

    # Print and save hyperparameters
    helpers.print_dictionary(hp_dict = vars(config) | vars(args), text = "Hyperparameters/settings:")
    pkl_path = os.path.join(config.pkl_dir, "hyperparams.pkl")
    with open(pkl_path, 'wb') as file:
        pickle.dump(vars(config) | vars(args), file)
    print()

    # ## Main Code

    # Retraining hyperparams
    if args.retrain:
        if not args.mlp_bottleneck_dim or not args.mlp_depth:
            raise Exception(f"Please specify an --mlp_bottleneck_dim ({args.mlp_bottleneck_dim}) and an --mlp_depth ({args.mlp_depth})")

        args.mlp_in_dim = args.shape[0]*args.shape[1]

    if args.subtract_mean:
        print("Subtracting mean from dataset")
        X_train = X_train - torch.mean(X_train, 0)

    # Generate or load data
    args.saved_file = f'{args.save_dir_name}_ae_output.pkl' if args.save_dir_name is not None else f'{args.dataset}_ae_output.pkl'
    args.pkl_path = os.path.join(config.pkl_dir, args.saved_file)
    if args.load_pickle:
        print(f"Loading saved AE pickle for \"{args.dataset}\".")
        if os.path.exists(args.pkl_path):
            with open(args.pkl_path, 'rb') as file:
                ae_output = pickle.load(file)
        else:
            raise Exception(f"Saved pickle file does not exist for \"{args.dataset}\".")
    elif args.retrain:
        print(f"Retraining AE on \"{args.dataset}\".")
        ae_output, train_losses, val_losses = helpers.train_AE(train_dataset=X_train,
                                val_dataset=X_val,
                                lr=args.lr,
                                iters=args.iters,
                                mlp_in_dim=args.mlp_in_dim,
                                mlp_bottleneck_dim=args.mlp_bottleneck_dim,
                                mlp_depth=args.mlp_depth,
                                spectral_lambda=args.spectral_lambda,
                                shape=args.shape,
                                batch_size=args.batch_size,
                                print_mod=args.print_mod,
                                fft_pos_freq=args.fft_pos_freq,
                                fft_normalize=args.fft_normalize,
                                fft_centering=args.fft_centering,
                                device=args.device,
                                save_model_spectrum=args.save_model_spectrum)
        if args.save_ae_output:
            with open(args.pkl_path, 'wb') as file:
                pickle.dump(ae_output, file)
            print()
        
        # Save pareto
        if args.save_pareto:
            print(f"Computing Pareto")
            optimal = plot_data.calculate_and_plot_pareto(train_losses, angle=-45, plot=True)

            pkl_path = os.path.join(config.pkl_dir, "pareto.pkl")
            with open(pkl_path, 'wb') as file:
                pickle.dump(optimal, file)

            pkl_path = os.path.join(config.pkl_dir, "train_losses.pkl")
            with open(pkl_path, 'wb') as file:
                pickle.dump(train_losses, file)

            pkl_path = os.path.join(config.pkl_dir, "val_losses.pkl")
            with open(pkl_path, 'wb') as file:
                pickle.dump(val_losses, file)

    if args.generate_integral_data or args.generate_2dfft_pdf or args.generate_fft_mse_pdf:
        print("Computing OG SVD.")

        # Get SVD of original dataset
        og_svd_pkl_path = Path(config.pkl_dir) / "og_svd.pkl"
        if og_svd_pkl_path.exists():
            with open(og_svd_pkl_path, 'rb') as file:
                og_svd = pickle.load(file)
        else:
            og_svd = torch.svd(X_train)
            with open(og_svd_pkl_path, 'wb') as file:
                pickle.dump(og_svd, file)

        print("Computing OG FFT.")

        # Get FFT of original dataset
        og_fft_pkl_path = Path(config.pkl_dir) / "og_fft.pkl"
        if og_fft_pkl_path.exists():
            with open(og_fft_pkl_path, 'rb') as file:
                og_fft = pickle.load(file)
        else:
            og_fft = helpers.calculate_ffts(og_svd, args.shape, args.fft_pos_freq, args.fft_normalize, args.fft_centering, args.crop)
            with open(og_fft_pkl_path, 'wb') as file:
                pickle.dump(og_fft, file)
        
        if args.retrain or args.load_pickle:
            # Get SVD of AE reconstruction of dataset
            print("Computing AE SVD.")
            ae_svd = torch.svd(torch.from_numpy(ae_output))

            print("Computing AE FFT.")
            ae_fft = helpers.calculate_ffts(ae_svd, args.shape, args.fft_pos_freq, args.fft_normalize, args.fft_centering, args.crop)

    if args.generate_integral_data or args.generate_2dfft_pdf:
        print("Generating integral data.")

        if args.retrain or args.load_pickle:
            zipped = zip([og_fft, ae_fft], ["og", "ae"])
        else:
            zipped = zip([og_fft], ["og"])

        for fft, name in zipped:
            filename = f'{args.dataset.lower()}_{name}_integral.pkl'
            integral_data = helpers.generate_integral_data(
                ffts=fft, filename=filename,
                p=args.p_value, smoothing=args.smoothing,
                radius_mult=args.radius_mult)

    if args.generate_eigenimage_pdf:
        print("Generating PDFs of eigenimages.")
        if args.retrain or args.load_pickle:
            plot_data.plot_several_image_samples(
                [og_svd[2], ae_svd[2]],
                ['OG', 'AE'],
                args.shape)
        else:
            plot_data.plot_several_image_samples(
                [og_svd[2]],
                ['OG'],
                args.shape)

    if args.generate_2dfft_pdf:
        print("Generating PDFs of 2DFFTs.")
        if args.retrain or args.load_pickle:
            plot_data.plot_2D_ffts(
                [og_fft, ae_fft],
                ['OG', 'AE'],
                integral_data=integral_data)
        else:
            plot_data.plot_2D_ffts(
                [og_fft],
                ['OG'],
                integral_data=integral_data)

    if args.generate_fft_mse_pdf:
        print("Generating PDFs MSE for 2DFFTs.")
        mse = np.mean((ae_fft - og_fft)**2,axis=(1,2))
        plot_data.plot_transition_frames(x=np.arange(mse.shape[0]), y=mse, label='MSE', title='MSE Per EigenImage', xaxis_title='EigenImage', yaxis_title='MSE', fname="transition_frame", save=True)
        pkl_path = os.path.join(config.pkl_dir, "fft_mse.pkl")
        with open(pkl_path, 'wb') as file:
            pickle.dump(mse, file)

    if args.generate_movies:
        print("Converting PDF into JPG.")

        # Convert all PDF into JPG
        cmd = f"cp {os.path.join(config.pdf_mass_dir,'*')} {os.path.join(config.jpg_mass_dir, '')} && cd {config.jpg_mass_dir} && mogrify -format jpg -quality 100 -density 200 *.pdf && rm *.pdf"
        print(f"Running {cmd}")
        os.system(cmd)

        print("Creating MP4s.")

        # Put all relevant JPG into a single video
        if args.generate_2dfft_pdf:
            mp4_fname = f"2DFFT_standard_{args.dataset}.mp4"
            cmd = f"cd {config.jpg_mass_dir} && ffmpeg -framerate {args.framerate} -pattern_type glob -i '2DFFT_*_standard.jpg' -pix_fmt yuv420p -c:v libx264 {mp4_fname}"
            print(f"Running {cmd}")
            os.system(cmd)
            cmd = f"mv {os.path.join(config.jpg_mass_dir, mp4_fname)} {config.mp4_dir}"
            print(f"Running {cmd}")
            os.system(cmd)

        if args.generate_eigenimage_pdf:
            mp4_fname = f"modes_{args.dataset}.mp4"
            cmd = f"cd {config.jpg_mass_dir} && ffmpeg -framerate {args.framerate} -pattern_type glob -i '*multiple*.jpg' -pix_fmt yuv420p -c:v libx264 {mp4_fname}"
            print(f"Running {cmd}")
            os.system(cmd)
            cmd = f"mv {os.path.join(config.jpg_mass_dir, mp4_fname)} {config.mp4_dir}"
            print(f"Running {cmd}")
            os.system(cmd)

    # Separate out everything into separate folders
    if args.generate_movies and args.keep_mp4s_only:
        cmd = f"rm -rf {config.jpg_mass_dir} {config.pdf_mass_dir}"
        print(f"Running {cmd}")
        #os.system(cmd)

    # Delete empty folders
    if not args.save_mass_dirs:
        for tmp_dir in [config.mp4_dir, config.jpg_mass_dir, config.jpg_dir, config.pdf_mass_dir, config.pdf_dir, config.pkl_dir]:
            cmd = f"if [ -z \"$( ls -A \'{tmp_dir}\' )\" ]; then rmdir {tmp_dir}; fi"
            print(f"Running {cmd}")
            #os.system(cmd)

if __name__ == '__main__':
    # To allow CLIs
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help="Dataset to run (MNIST, FMNIST, or YaleFaces)")
    parser.add_argument('--device', type=str, default="cuda:2", help="Which device to run on (cuda:0, cuda:1, cuda:2)")
    parser.add_argument('--generate_movies', default=False, action='store_true', help="Generate MP4 videos")
    parser.add_argument('--mlp_depth', type=int, default=None, help="MLP model depth (number of layers)")
    parser.add_argument('--mlp_bottleneck_dim', type=int, default=None, help="MLP model bottleneck size")
    parser.add_argument('--generate_2dfft_pdf', default=False, action='store_true', help="Generate PDFs of 2DFFTs")
    parser.add_argument('--generate_eigenimage_pdf', default=False, action='store_true', help="Generate PDFs of Eigenimages")
    parser.add_argument('--generate_fft_mse_pdf', default=False, action='store_true', help="Generate PDFs of MSE for FFTs")
    parser.add_argument('--retrain', default=False, action='store_true', help="Retrain AutoEncoder")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for retraining")
    parser.add_argument('--iters', type=int, default=500, help="Number of epochs for retraining")
    parser.add_argument('--load_pickle', default=False, action='store_true', help="Load Saved AutoEncoder Frames")
    parser.add_argument('--generate_integral_data', default=False, action='store_true', help="Generate integral data for PySindy")
    parser.add_argument('--keep_mp4s_only', default=False, action='store_true', help="When generating movies, remove intermediate files and pickles to save space")
    parser.add_argument('--framerate', type=int, default=24, help="Generated movie framerate")
    parser.add_argument('--subtract_mean', default=False, action='store_true', help="Subtract mean image from dataset")
    parser.add_argument('--crop', default=False, action='store_true', help="Crop to middle 1/3 of image along x and y axis when generating plots")
    parser.add_argument('--spectral_lambda', type=float, default=0.0, help="Regularization value for spectral loss")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size (enter -1 for full dataset)")
    parser.add_argument('--print_mod', type=int, default=1, help="Every n prints when training")
    parser.add_argument('--custom_p_value', type=float, default=None, help="Override p-value for radial integrals")
    parser.add_argument('--custom_smoothing', type=float, default=None, help="Override smoothing for radial integrals")
    parser.add_argument('--custom_radius_mult', type=float, default=None, help="Override radius mult for radial integrals")
    parser.add_argument('--save_dir_name', type=str, default=None, help="Folder where to save run data and name of pickle file, default is just dataset name")
    parser.add_argument('--val_split', type=float, default=0.0, help="Percentage of training data to use for validation.")
    parser.add_argument('--save_ae_output', default=False, action='store_true', help="Save autoencoder output as a pkl file")
    parser.add_argument('--save_pareto', default=False, action='store_true', help="Save pareto optimal point as a pkl file")
    parser.add_argument('--save_mass_dirs', default=False, action='store_true', help="Save mass file directories")
    parser.add_argument('--save_model_spectrum', default=False, action='store_true', help="Save learned MLP")
    args = parser.parse_args()

    main(args)