""" script for training a Self Attention GAN on celeba images """

import argparse

import numpy as np
import torch as th
from torch.backends import cudnn

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# enable fast training
cudnn.benchmark = True

# set seed = 3
th.manual_seed(seed=3)


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_file", action="store", type=str,
                        default=None,
                        help="pretrained weights file for generator")

    parser.add_argument("--generator_optim_file", action="store", type=str,
                        default=None,
                        help="saved state for generator optimizer")

    parser.add_argument("--discriminator_file", action="store", type=str,
                        default=None,
                        help="pretrained_weights file for discriminator")

    parser.add_argument("--discriminator_optim_file", action="store", type=str,
                        default=None,
                        help="saved state for discriminator optimizer")

    parser.add_argument("--images_dir", action="store", type=str,
                        default="../data/celeba",
                        help="path for the images directory")

    parser.add_argument("--folder_distributed", action="store", type=bool,
                        default=False,
                        help="whether the images directory contains folders or not")

    parser.add_argument("--sample_dir", action="store", type=str,
                        default="samples/1/",
                        help="path for the generated samples directory")

    parser.add_argument("--model_dir", action="store", type=str,
                        default="models/1/",
                        help="path for saved models directory")

    parser.add_argument("--loss_function", action="store", type=str,
                        default="relativistic-hinge",
                        help="loss function to be used: 'hinge', 'relativistic-hinge', " +
                             "'lsgan' or 'standard-gan'")

    parser.add_argument("--depth", action="store", type=int,
                        default=6,
                        help="Depth of the GAN")

    parser.add_argument("--latent_size", action="store", type=int,
                        default=256,
                        help="latent size for the generator")

    parser.add_argument("--batch_size", action="store", type=int,
                        default=24,
                        help="batch_size for training")

    parser.add_argument("--start", action="store", type=int,
                        default=1,
                        help="starting epoch number")

    parser.add_argument("--num_epochs", action="store", type=int,
                        default=3,
                        help="number of epochs for training")

    parser.add_argument("--feedback_factor", action="store", type=int,
                        default=100,
                        help="number of logs to generate per epoch")

    parser.add_argument("--num_samples", action="store", type=int,
                        default=64,
                        help="number of samples to generate for creating the grid" +
                             " should be a square number preferably")

    parser.add_argument("--gen_coord_conv", action="store", type=bool,
                        default=True,
                        help="Whether to use CoordConv in generator")

    parser.add_argument("--gen_upsampling", action="store", type=bool,
                        default=False,
                        help="Whether to use Upsampling in generator")

    parser.add_argument("--dis_coord_conv", action="store", type=bool,
                        default=True,
                        help="Whether to use CoordConv in discriminator")

    parser.add_argument("--checkpoint_factor", action="store", type=int,
                        default=1,
                        help="save model per n epochs")

    parser.add_argument("--g_lr", action="store", type=float,
                        default=0.003,
                        help="learning rate for generator")

    parser.add_argument("--d_lr", action="store", type=float,
                        default=0.0003,
                        help="learning rate for discriminator")

    parser.add_argument("--adam_beta1", action="store", type=float,
                        default=0,
                        help="value of beta_1 for adam optimizer")

    parser.add_argument("--adam_beta2", action="store", type=float,
                        default=0.99,
                        help="value of beta_2 for adam optimizer")

    parser.add_argument("--use_spectral_norm", action="store", type=bool,
                        default=True,
                        help="Whether to use spectral normalization or not")

    parser.add_argument("--data_percentage", action="store", type=float,
                        default=100,
                        help="percentage of data to use")

    parser.add_argument("--num_workers", action="store", type=int,
                        default=3,
                        help="number of parallel workers for reading files")

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """
    from CMSG_GAN.GAN import CMSG_GAN
    from data_processing.DataLoader import FlatDirectoryImageDataset, \
        get_transform, get_data_loader, FoldersDistributedDataset
    from CMSG_GAN.Losses import HingeGAN, RelativisticAverageHingeGAN, \
        StandardGAN, LSGAN

    # create a data source:
    data_source = FlatDirectoryImageDataset if not args.folder_distributed \
        else FoldersDistributedDataset

    dataset = data_source(
        args.images_dir,
        transform=get_transform((int(np.power(2, args.depth + 1)),
                                 int(np.power(2, args.depth + 1)))))

    data = get_data_loader(dataset, args.batch_size, args.num_workers)

    # create a gan from these
    cmsg_gan = CMSG_GAN(depth=args.depth,
                        latent_size=args.latent_size,
                        dis_coord_conv=args.dis_coord_conv,
                        gen_coord_conv=args.gen_coord_conv,
                        use_spectral_norm=args.use_spectral_norm,
                        use_upsampling=args.gen_upsampling,
                        device=device)

    if args.generator_file is not None:
        # load the weights into generator
        cmsg_gan.gen.load_state_dict(th.load(args.generator_file))

    print("Generator Configuration: ")
    print(cmsg_gan.gen)

    if args.discriminator_file is not None:
        # load the weights into discriminator
        cmsg_gan.dis.load_state_dict(th.load(args.discriminator_file))

    print("Discriminator Configuration: ")
    print(cmsg_gan.dis)

    # create optimizer for generator:
    gen_optim = th.optim.Adam(cmsg_gan.gen.parameters(), args.g_lr,
                              [args.adam_beta1, args.adam_beta2])

    dis_optim = th.optim.Adam(cmsg_gan.dis.parameters(), args.d_lr,
                              [args.adam_beta1, args.adam_beta2])

    if args.generator_optim_file is not None:
        gen_optim.load_state_dict(th.load(args.generator_optim_file))

    if args.discriminator_optim_file is not None:
        dis_optim.load_state_dict(th.load(args.discriminator_optim_file))

    loss_name = args.loss_function.lower()

    if loss_name == "hinge":
        loss = HingeGAN
    elif loss_name == "relativistic-hinge":
        loss = RelativisticAverageHingeGAN
    elif loss_name == "standard-gan":
        loss = StandardGAN
    elif loss_name == "lsgan":
        loss = LSGAN
    else:
        raise Exception("Unknown loss function requested")

    # train the GAN
    cmsg_gan.train(
        data,
        gen_optim,
        dis_optim,
        loss_fn=loss(device, cmsg_gan.dis),
        num_epochs=args.num_epochs,
        checkpoint_factor=args.checkpoint_factor,
        data_percentage=args.data_percentage,
        feedback_factor=args.feedback_factor,
        num_samples=args.num_samples,
        sample_dir=args.sample_dir,
        save_dir=args.model_dir,
        log_dir=args.model_dir,
        start=args.start
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
