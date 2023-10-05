# gitにadd , commit , pushまでを一括で行う
# ただしcommitのコメントは引数で指定する
# 引数は-mで指定する

# 引数の数をチェック
if [ $# -ne 2 ]; then
    echo "引数の数が間違っています"
    exit 1
fi

# 引数のチェック
if [ $1 != "-m" ]; then
    echo "引数が間違っています"
    exit 1
fi

# git add
git add .

# git commit
git commit -m $2

# git push
git push origin main