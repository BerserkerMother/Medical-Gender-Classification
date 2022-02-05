mkdir data/features/IXI -p
wget https://download1501.mediafire.com/6shl1pivv59g/evhdoju6ins6rv0/IXI_Train_Part1.zip -P data/features
wget https://download848.mediafire.com/d0eq13q0dffg/hsh1em2udn43p0m/IXI_Train_Part2.zip -P data/features
wget http://download1525.mediafire.com/1x75qlc9v3yg/1t0xt48tufqfduh/OASIS.zip -P data/features
unzip data/features/IXI_Train_Part1.zip -d data/features/IXI
unzip data/features/IXI_Train_Part2.zip -d data/features/IXI
unzip data/features/OASIS.zip -d data/features/

cp data/features/IXI/IXI_Train_Part1/* data/features/IXI && rm data/features/IXI/IXI_Train_Part1
cp data/features/IXI/IXI_Train_Part2/* data/features/IXI && rm data/features/IXI/IXI_Train_Part2


mkdir data/annotations -p
tar xvzf annotations.tar.gz -C data/annotations
