## train_and_eval
# deu.rst.pcc, eng.dep.scidtb, eng.pdtb.pdtb, eng.rst.gum, eng.rst.rstdt, eng.sdrt.stac, eus.rst.ert, fas.rst.prstc, 
# fra.sdrt.annodis, ita.pdtb.luna, nld.rst.nldt, por.pdtb.crpc, por.rst.cstn, rus.rst.rrt, spa.rst.rststb, spa.rst.sctb, 
# tha.pdtb.tdtb tur.pdtb.tdb zho.dep.scidtb, zho.pdtb.cdtb, zho.rst.gcdt, zho.rst.sctb

## eval only
# eng.dep.covdtb, eng.pdtb.tedm, por.pdtb.tedm, tur.pdtb.tedm
# <<"COMMENT"
python3 task3.py --do_dev --do_test \
                 --model_type="base" \
                 --dataset="super.sdrt" \
                 --feature_size=0 \
                 --max_seq_length=256 \
                 # --train_batch_size=16 \
                 # --learning_rate=1e-4 \
                 # --do_adv \

# COMMENT
# eng.pdtb.pdtb


<<"COMMENT"
# for dataset in deu.rst.pcc eng.dep.scidtb eng.rst.gum eng.sdrt.stac eus.rst.ert fas.rst.prstc fra.sdrt.annodis ita.pdtb.luna nld.rst.nldt por.rst.cstn rus.rst.rrt spa.rst.rststb spa.rst.sctb tur.pdtb.tdb zho.dep.scidtb zho.pdtb.cdtb zho.rst.gcdt zho.rst.sctb
# for dataset in eng.pdtb.pdtb eng.rst.rstdt eus.rst.ert fra.sdrt.annodis por.pdtb.crpc tha.pdtb.tdtb zho.rst.gcdt 
# for dataset in eng.rst.rstdt eus.rst.ert fra.sdrt.annodis tha.pdtb.tdtb
# for dataset in eng.pdtb.pdtb por.rst.cstn zho.rst.gcdt tha.pdtb.tdtb
for dataset in eng.rst.rstdt
do
    python3 task3.py --do_train --do_adv \
                     --dataset=${dataset} \
                     --max_seq_length=256 \
                     --feature_size=0

done
COMMENT
