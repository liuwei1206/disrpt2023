
## 1.for corpora with a training set
for dataset in deu.rst.pcc eng.rst.gum eng.rst.gum eng.rst.rstdt eus.rst.ert fas.rst.prstc nld.rst.nldt por.rst.cstn rus.rst.rrt spa.rst.rststb spa.rst.sctb zho.rst.gcdt zho.rst.sctb zho.dep.scidtb eng.dep.scidtb eng.pdtb.pdtb ita.pdtb.luna por.pdtb.crpc tha.pdtb.tdtb tur.pdtb.tdb zho.pdtb.cdtb
do
    python3 task3.py --do_train \
                     --dataset=${dataset}
done

## 2. for corpora without a training set
for dataset in eng.dep.covdtb eng.pdtb.tedm por.pdtb.tedm tur.pdtb.tedm
do
    python3 task3.py --do_dev --do_test \
                     --dataset=${dataset}
done
