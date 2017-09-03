N=40000

Ntotal=40000

L=4

Nc=$( echo " $Ntotal/$N " |bc)

#training set
python w_zbasis.py $L $N 
 
touch trainD.txt ; rm trainD.txt ; touch trainD.txt 

for (( c=1; c<=$Nc; c++ ))
do  
  cat train.txt  >>trainD.txt 
done

rm train.txt 


# test set
Ntotal=10000

Nc=1 

python w_zbasis.py $L $Ntotal

touch test.txt ; rm test.txt ; touch test.txt

for (( c=1; c<=$Nc; c++ ))
do
  cat train.txt  >>test.txt
done

rm train.txt


