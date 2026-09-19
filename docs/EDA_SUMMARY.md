# Exploratory Data Analysis

## Dataset

The working dataset contains 232,074 labeled text samples.

| Class | Count | Share |
| --- | ---: | ---: |
| Positive class | 116,037 | 50.0% |
| Negative class | 116,037 | 50.0% |
| Total | 232,074 | 100.0% |

No rows were dropped during the final cleaning pass used for this analysis.

## Text length

- mean length: 131.9 words
- median length: 60.0 words
- 63.7% of samples fit within 128 tokens
- 80.7% fit within 256 tokens

The distribution is right-skewed, with many short samples and a smaller number of long texts.

## Configuration decisions

### Sequence length

The training configuration uses a maximum sequence length of 256. This covers 80.7% of samples without the computational cost of using 512 tokens for every example.

### Class weighting

The dataset is evenly split between the two classes, so the current configuration does not apply class weights.

### Data split

The project stores fixed split indices so repeated experiments use the same train, validation, and test examples.

## Current preprocessing configuration

```yaml
preprocessing:
  max_seq_length: 256
  padding: true
  truncation: true

class_weights:
  enabled: false
```

## Notes

The training dataset is not committed to this repository. Only the saved split indices and experiment outputs are versioned.
