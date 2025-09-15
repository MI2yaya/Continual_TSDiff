#!/usr/bin/env python3
"""
List available GluonTS datasets with basic metadata
"""

from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
import logging

# Suppress verbose logging
logging.getLogger("gluonts").setLevel(logging.WARNING)

def list_datasets():
    """List all available GluonTS datasets"""
    dataset_names = sorted(dataset_recipes.keys())
    
    print(f"Available GluonTS datasets ({len(dataset_names)} total):")
    print("=" * 80)
    
    for name in dataset_names:
        try:
            dataset = get_dataset(name, regenerate=False)
            sample = next(iter(dataset.train))
            
            print(f"{name}")
            print(f"  Frequency: {dataset.metadata.freq}")
            print(f"  Prediction length: {dataset.metadata.prediction_length}")
            print(f"  Train series: {len(list(dataset.train))}")
            print(f"  Sample length: {len(sample['target'])}")
            print()
            
        except Exception as e:
            print(f"{name} - Error: {str(e)[:50]}...")
            print()

def find_similar_groups():
    """Find potentially similar dataset groups"""
    dataset_names = sorted(dataset_recipes.keys())
    
    print("\nPotentially similar groups for continual learning:")
    print("=" * 60)
    
    # Energy related
    energy = [name for name in dataset_names if any(k in name.lower() for k in ['solar', 'electricity', 'energy'])]
    if energy:
        print(f"Energy: {energy}")
    
    # Transportation
    transport = [name for name in dataset_names if any(k in name.lower() for k in ['traffic', 'uber', 'taxi', 'pedestrian'])]
    if transport:
        print(f"Transportation: {transport}")
    
    # Financial
    financial = [name for name in dataset_names if any(k in name.lower() for k in ['exchange', 'm4', 'stock'])]
    if financial:
        print(f"Financial: {financial}")

if __name__ == "__main__":
    list_datasets()
    find_similar_groups()

