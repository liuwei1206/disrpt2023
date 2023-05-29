
## 2. for corpora without a training set
for dataset in eng.pdtb.pdtb eng.rst.gum eng.rst.rstdt
do
    python3 task3.py --do_dev --do_test \
                     --dataset=${dataset}
done
