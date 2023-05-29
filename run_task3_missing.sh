## 1.for corpora with a training set
for dataset in eng.sdrt.stac fra.sdrt.annodis eng.pdtb.pdtb eng.rst.gum eng.rst.rstdt
do
    python3 task3.py --do_train \
                     --dataset=${dataset}
done

