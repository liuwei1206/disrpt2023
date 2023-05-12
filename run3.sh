## train_and_eval
# deu.rst.pcc, eng.dep.scidtb, eng.pdtb.pdtb, eng.rst.gum, eng.rst.rstdt, eng.sdrt.stac, eus.rst.ert, fas.rst.prstc, 
# fra.sdrt.annodis, ita.pdtb.luna, nld.rst.nldt, por.pdtb.crpc, por.rst.cstn, rus.rst.rrt, spa.rst.rststb, spa.rst.sctb, 
# tha.pdtb.tdtb tur.pdtb.tdb zho.dep.scidtb, zho.pdtb.cdtb, zho.rst.gcdt, zho.rst.sctb

## eval only
# eng.dep.covdtb, eng.pdtb.tedm, por.pdtb.tedm, tur.pdtb.tedm
# <<"COMMENT"
python3 task3_submit.py --do_dev --do_test \
                        --model_type="base" \
                        --dataset="eng.pdtb.pdtb" \
                        --max_seq_length=256 \
                        # --train_batch_size=16 \
                        # --learning_rate=1e-4 \
                        # --do_adv \

# COMMENT
# eng.pdtb.pdtb
