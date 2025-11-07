<!-- pip install --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126 -->
pip install --force-reinstall --no-deps git+https://github.com/unslothai/unsloth git+https://github.com/unslothai/unsloth-zoo
pip install -r requirements.txt
pip install ninja
pip install xformers --index-url https://download.pytorch.org/whl/cu128