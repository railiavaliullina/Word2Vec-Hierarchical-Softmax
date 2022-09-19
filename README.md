# Word2Vec-Hierarchical-Softmax

## About The Project

1) Implementation of Skip-gram + Hierarchical Softmax (using NumPy + Cython),
2) Training on Text8 corpus (100 mb of Wiki data)

## Getting Started


File to run:


## Implementation details

The following modules have been implemented:

  - Word to index dictionary
  
  - Index to word dictionary
  
  - List of word counts (with the same index as in index to word dic)
  
  - Built huffman tree
  
  - Additional structures to quickly get paths and nodes according to Huffman
  
  - Min word count 5
  
 
 ## Training details
 
- Text8 corpus (100 mb of Wiki data)

- Calculate Huffman tree, basic structures
noise distribution

- W_input (vocab_size x 200) with Xavier init, W_output (200 x vocab_size-1) with zeros

- Linear moving across the subsampling corpus with batch size on your choice

- Get context of size C = 5 and put in batch 2*C context pairs with the same target word

- Follow Hierarchical Softmax algo
  
  
  
  
  
 
