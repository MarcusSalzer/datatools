. sh/test.sh
if [ $? -eq 1 ]; then
    return 1
fi
pip install . --upgrade