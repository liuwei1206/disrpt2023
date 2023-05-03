
if [ -d data/embeddings ]
then
    echo "data/embeddings exists"
else
    echo "new"
    mkdir data/embeddings
fi

for lang in de fr en # fa eu it nl pt ru es zh tr en
do
    if [ -f data/embeddings/cc.${lang}.300.bin ]
    then 
        echo "cc.${lang}.300.bin exists"
    else
        echo "new"
        wget -O data/embeddings/cc.${lang}.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.${lang}.300.bin.gz
        gunzip data/embeddings/cc.${lang}.300.bin.gz
    fi
done
