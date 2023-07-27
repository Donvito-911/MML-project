"""Main module"""
from hparam_explorer import HparamExplorer
from discriminator_builder import  discriminator1, discriminator2
from generator_builder import generator1, generator2

if __name__ == "__main__":
    hparam_explorer = HparamExplorer(
        train_file="data/processed_data.h5",
        latent_dims=[128, 256],
        batch_size=32,
        n_epochs=300,
        generators=[generator1, generator2],
        discriminators=[discriminator1, discriminator2]
    )
    
    hparam_explorer.run()