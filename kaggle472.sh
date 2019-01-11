files=("30_22" "34_30" "35_36" "36_40" "37_33" "41_32")

for f in "${files[@]}"
do
    wget http://homepage.ntu.edu.tw/~b04611017/final/test$f.pt
    python3 test.py test$f
done
python3 ensemble.py
