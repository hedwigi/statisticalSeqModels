# A comparison of HMM/MEMM/CRF

## Environment
python 3
nltk
numpy

## Reproducing results

- run main.py to get predicted results of HMM or MEMM
- run run_hmmem.py to get predicted results of HMM(em)
- run crf++ with command lines, with the train/test files in data/; predicted results can be saved in data/pred_results/
- user scripts/conlleval_rev.pl to evaluate files in data/pred_results/

## Preparing new data
- put new data in data/orig/ with format:
```
不/d 忘/v 藏北/s 人民/n 的/u 拉萨/ns 市民/n （/w 图片/n ）/w
```
- run data/transform2conll.py to transform data/orig files to seg/ner/pos files with crf++ format 


## References
- blog link (to be added)
- https://github.com/tostq/Easy_HMM
- https://spaces.ac.cn/archives/3922
- https://github.com/yh1008/MEMM
- https://cocoxu.github.io/courses/5525_slides_spring17/15_more_memm.pdf
- https://taku910.github.io/crfpp/
