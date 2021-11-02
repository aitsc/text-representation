# Introduction
About datasets for information retrieval and recommendation tasks

## Sources

arXiv<sup>[1]</sup>, DBLP<sup>[2]</sup>, USPTO<sup>[3]</sup>

[1] ftp://3lib.org//oai\_dc/arxiv

[2] https://aminer.org/citation

[3] http://www.patentsview.org/download

## Extraction

We randomly selected 1000 papers to be queried and 20,000 candidate papers from the arXiv dataset.

We randomly select 1000 papers to be recommended and 20,000 candidate papers from the DBLP dataset.

We randomly select 1000 patents to be recommended and 20,000 candidate patents from the USPTO dataset.

## Preprocessing

1. Convert all text to lowercase;
2. Remove HTML labels;
3. Restore HTML escape characters;
4. Split text with punctuation;
5. Remove tokens without letters. 

## Result

arXiv20000

|                 | Query papers | Candidate papers |
| --------------- | ------------ | ---------------- |
| Papers          | 1000         | 20000            |
| Years           | 2016         | 1991-2015        |
| Words           | 205-418      | 205-425          |
| Classifications | 146          | 331              |
| Candidates      | 22-95        | -                |

DBLP20000

|           | Recommended papers | Candidate papers |
| --------- | ------------------ | ---------------- |
| Papers    | 1000               | 20000            |
| Years     | 2017               | 1967-2016        |
| Words     | 156-412            | 155-421          |
| Citations | 15-16              | -                |

USPTO20000

|           | Recommended patents | Candidate patents |
| --------- | ------------------- | ----------------- |
| Patents   | 1000                | 20000             |
| Years     | 2017-2018           | 2002-2016         |
| Words     | 255-629             | 250-680           |
| Citations | 8-29                | -                 |

## Format

First line: {'text ID of candidates or citations':{'text ID of papers or patents',..},..}

Other line: text ID of papers or patents \t title \t abstract