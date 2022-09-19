set -x
cd ..

python test_bert.py --AT_type FGM --use_attack 1  --epoch 32

#echo "train FreeAT"
#python test_bert.py --AT_type FreeAT --use_attack 1  --epoch 32
