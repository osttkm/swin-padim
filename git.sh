# gitにadd , commit , pushまでを一括で行う
# ただしcommitのコメントは引数で指定する

# 引数の数をチェック
if [ $# -ne 1 ]; then
    echo "引数の数が間違っています"
    exit 1
fi

# git add
git add .

# git commit
git commit -m "$1"

# git push
git push origin main
