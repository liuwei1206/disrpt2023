
for dataset in deu.rst.pcc eng.rst.rstdt eus.rst.ert fas.rst.prstc nld.rst.nldt por.rst.cstn spa.rst.rststb spa.rst.sctb zho.rst.sctb zho.rst.gcdt eng.pdtb.pdtb ita.pdtb.luna por.pdtb.crpc tha.pdtb.tdtb zho.pdtb.cdtb eng.sdrt.stac fra.sdrt.annodis eng.dep.scidtb zho.dep.scidtb
do
    python3 task12.py --do_train \
                     --dataset=${dataset} \
                     --max_seq_length=512 \
                     --train_batch_size=16 \
                     --eval_batch_size=32 \
                     --learning_rate=3e-5 \
                     --dropout=0.1 \
                     --num_train_epochs=10 \
                     --model_type="bilstm+crf"
done


#for dataset in deu.rst.pcc eng.dep.scidtb eng.rst.gum eng.sdrt.stac eus.rst.ert fas.rst.prstc fra.sdrt.annodis nld.rst.nldt por.rst.cstn por.pdtb.crpc rus.rst.rrt spa.rst.rststb spa.rst.sctb zho.dep.scidtb zho.rst.gcdt zho.rst.sctb tha.pdtb.tdtb
#for dataset in rus.rst.rrt
for dataset in eng.rst.gum rus.rst.rrt tur.pdtb.tdb
do
    python3 task12.py --do_train \
                     --dataset=${dataset} \
                     --max_seq_length=512 \
                     --train_batch_size=16 \
                     --eval_batch_size=32 \
                     --learning_rate=1e-5 \
                     --dropout=0.1 \
                     --num_train_epochs=10 \
                     --model_type="bilstm+crf"
done



for dataset in eng.pdtb.tedm por.pdtb.tedm tur.pdtb.tedm eng.dep.covdtb
do
    python3 task12.py --do_dev \
                     --do_test \
                     --model_type="bilstm+crf" \
                     --trained_model="eng.pdtb.pdtb" \
                     --dataset=${dataset} \
                     --max_seq_length=512 \
                     --train_batch_size=16 \
                     --eval_batch_size=32 \
                     --learning_rate=1e-5 \
                     --dropout=0.1 \
                     --num_train_epochs=1 \
                     --model_type="bilstm+crf"
done
