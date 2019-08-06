# Geotagging Twitter Users

Geotagging Twitter Users is a location estimation project based on twitter users.

## Installation
- Requires Python3

- Clone git repo

`git clone https://github.com/maymayan/geotagging_twitter`

- Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.
 
`pip install -r requirements.txt`

## Usage

**Predict Locations**

`python3 predict_locations.py <input_file_path> <output_file_path>`

**Input File Format**

Input file should be a CSV that has 4 columns: nickname, bio, tweet and hashtag
For a row in the input file, tweet column should contain space separated tweets sent by the given username and hashtag column should contain space separated hashtags of tweets sent by the given username

Input file should contain the following header:
`nickname,bio,tweet,hashtag`

A sample input file can be found in `data/test.csv`

```
nickname,bio,tweet,hashtag
.	.	.	.	.	.	.
.	.	.	.	.	.	.
.	.	.	.	.	.	.
.	.	.	.	.	.	.
.	.	.	.	.	.	.
```

**Output File Format**

Output file contains location predictions corresponding to rows in the input file.