# deu.rst.pcc, eng.dep.scidtb, eng.pdtb.pdtb, eng.rst.gum, eng.rst.rstdt, eng.sdrt.stac, eus.rst.ert, fas.rst.prstc, 
# fra.sdrt.annodis, ita.pdtb.luna, nld.rst.nldt, por.rst.cstn, rus.rst.rrt, spa.rst.rststb, spa.rst.sctb, tur.pdtb.tdb
# zho.dep.scidtb, zho.pdtb.cdtb, zho.rst.gcdt, zho.rst.sctb
# <<"COMMENT"
python3 task3.py --do_train \
                 --dataset="zho.rst.sctb" \
                 --max_seq_length=384 \
                 --train_batch_size=16 \
                 --eval_batch_size=32 \
                 --learning_rate=1e-4 \
                 --dropout=0.1 \
                 --num_train_epochs=10 \

# COMMENT
# eng.pdtb.pdtb


<<"COMMENT"
for dataset in eng.dep.scidtb eng.pdtb.pdtb eng.rst.gum eng.rst.rstdt eng.sdrt.stac eus.rst.ert fas.rst.prstc fra.sdrt.annodis nld.rst.nldt por.rst.cstn rus.rst.rr spa.rst.rststb spa.rst.sctb tur.pdtb.tdb
do
    python3 task3.py --do_train \
                     --dataset=${dataset} \
                     --max_seq_length=384 \
                     --train_batch_size=16 \
                     --eval_batch_size=32 \
                     --learning_rate=3e-5 \
                     --num_train_epochs=5
done
COMMENT
