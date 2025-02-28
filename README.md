# GPT2ish

A simple implementation of the GPT-2 architecture.

## Overview

GPT2ish is a Python-based project that provides a straightforward and customizable implementation of OpenAI's GPT-2 model. This repository aims to help users understand and experiment with the GPT-2 architecture by offering a clear and concise codebase.

## Model
The model can be found in modules/transformers.py

## Training
An example of the training can be found in training/warm_up/training.py where it was trained on the [tiny stories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. 

## Testing
An example of generating text in probabilistic manner (temperature) can be found in testing/warm_up/testing.py, here is an example from a short training session with 10000 stories, 8 heads, dk=16, dm=128, 4 layers and 100 epochs,

Prompt: "Once upon a time in an enchanted"

Generated story: "Once upon a time in an enchanted forest, there was a little boat. The boat sailed every day swimming in the water, swimming in the water.
One day, a little girl saw the most beautiful boat and thought it was so beautiful. She wanted to pick it up a plastic, so it flapped as it fluttered into'.
The little girl was so excited. She wanted to try it out of the water, but the little girl was too heavy for her to her parents."

😆 okay it's not the best, but this was just to test.
