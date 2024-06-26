import argparse
import os
import subprocess

def create_new_sh_content(args):
    new_sh_content = f'''#!/usr/bin/env bash
    #SBATCH --job-name=experiment
    #SBATCH --output=experiment%j.log
    #SBATCH --error=experiment%j.err
    #SBATCH --mail-user=liang@uni-hildesheim.de
    #SBATCH --mail-type=ALL
    #SBATCH --partition=STUD
    #SBATCH --gres=gpu:1

    source activate toyEnv
    srun python exp.py \\
        --dataset {args.dataset} \\
        --entropy_threshold {args.entropy_threshold} \\
        --run_epochs {args.run_epochs} \\
        --candidate_start_epoch {args.candidate_start_epoch} \\
        --tensorboard_comment "{args.tensorboard_comment}" \\
        --lr {args.lr} \\
        --l2 {args.l2} \\
        --batch_size {args.batch_size} \\
        {"--reduce_dataset" if args.reduce_dataset else ""} \\
        {"--accumulation_steps" if args.accumulation_steps else ""} \\
        {"--augmentation_type "+ args.augmentation_type if args.augmentation_type else ""}\\
        {"--simpleAugmentation_name " + str(args.simpleAugmentation_name) if args.simpleAugmentation_name else ""}\\
        {"--k_epoch_sampleSelection " + str(args.k_epoch_sampleSelection) if args.k_epoch_sampleSelection else ""}\\
        {"--augmente_epochs_list " + args.augmente_epochs_list if args.augmente_epochs_list else ""}\\
        {"--vae_accumulationSteps " + str(args.vae_accumulationSteps) if args.vae_accumulationSteps else ""}\\
        {"--vae_trainEpochs " + str(args.vae_trainEpochs) if args.vae_trainEpochs else ""}\\
        {"--GANs_trainEpochs " + str(args.GANs_trainEpochs) if args.GANs_trainEpochs else ""}\\
        {"--GANs_latentDim " + str(args.GANs_latentDim) if args.GANs_latentDim else ""}\\
        {"--GANs_lr " + str(args.GANs_lr) if args.GANs_lr else ""}\\
        {"--GANs_tensorboardComment " + str(args.GANs_tensorboardComment) if args.GANs_tensorboardComment else ""}\\
    '''
    return new_sh_content

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create new sh file for experiment')

    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=("MNIST", "CIFAR10", "FashionMNIST", "SVHN", "Flowers102", "Food101"), help='Dataset name')
    parser.add_argument('--entropy_threshold', type=float, default=0.5, help='Entropy threshold')
    parser.add_argument('--run_epochs', type=int, default=5, help='Number of epochs to run')
    parser.add_argument('--candidate_start_epoch', type=int, default=0, help='Epoch to start selecting candidates. Candidate calculation begind after the mentioned epoch')
    parser.add_argument('--tensorboard_comment', type=str, default='test_run', help='Comment to append to tensorboard logs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--reduce_dataset', action='store_true', help='Reduce the dataset size (for testing purposes only)')
    parser.add_argument('--accumulation_steps', type=int, default=None, help='Number of accumulation steps')
    
    parser.add_argument('--augmentation_type', type=str, default=None, choices=("vae", "simple", "GANs"), help='Augmentation type')
    parser.add_argument('--simpleAugmentation_name', type=str, default=None, choices=("random_color", "center_crop", "gaussian_blur", 
                                                                                    "elastic_transform", "random_perspective", "random_resized_crop", 
                                                                                    "random_invert", "random_posterize", "rand_augment", "augmix"), help='Simple Augmentation name')
    parser.add_argument('--k_epoch_sampleSelection', type=int, default=3, help='Number of epochs to select the common candidates')
    parser.add_argument('--augmente_epochs_list', type=list, default=None, help='certain epoch to augmente the dataset')

    parser.add_argument('--vae_accumulationSteps', type=int, default=4, help='Accumulation steps for VAE training')
    parser.add_argument('--vae_trainEpochs', type=int, default=100, help='Number of epochs to train vae')

    parser.add_argument('--GANs_trainEpochs', type=int, default=10, help='Number of epochs to train GANs')
    parser.add_argument('--GANs_latentDim', type=int, default=100, help='latent dim for GANs')
    parser.add_argument('--GANs_lr', type=float, default=0.001, help='learning rate for GANs')
    parser.add_argument('--GANs_tensorboardComment', type=str, default='debug with GANs', help='tensorboard comment for GANs')

    args = parser.parse_args() 

    new_sh_content = create_new_sh_content(args)
    with open('new_testScript.sh', 'w') as file:
        file.write(new_sh_content)
    print(f"Updated the content of 'new_testScript.sh'")




