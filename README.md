# ResNet Animal Classifier

A residual CNN written from scratch in PyTorch, trained to classify 15 animal species, and served through a small Flask app with an upload page.

## Classes

Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra

## Model

`src/models.py` defines `ResNet50`: a 7x7 stem, four residual stages with `[3, 4, 6, 3]` blocks (64/128/256/512 channels), global average pooling and a linear head. Each block is a two-layer 3x3 *basic* block. That gives the stage layout of ResNet-34, not the bottleneck blocks of the original ResNet-50, even though the class is called `ResNet50`.

Training (`main.ipynb`, `src/train.py`):

- Adam, lr `1e-5`, batch size 128, 50 epochs
- Cross-entropy plus an L2 penalty (`l2_lambda=0.001`) added to the loss by hand
- TensorBoard logging to `runs/`
- 70/15/15 random split of `src/data/animal_data/` (`Animals.train_test_val_data`)

The trained weights are committed as `model.pth`.

## Project structure

```
app.py               Flask app: GET / (upload page), POST /classify
main.ipynb           Training notebook
model.pth            Trained weights (state_dict)
src/
  dataset.py         Animals dataset (one folder per class) and train/val/test split
  models.py          Residual network
  train.py           Train/test loop with TensorBoard logging
  transforms.py      ToTensor + Normalize (optional flip / color distortion)
  helper.py          Accuracy, mean/std and timing helpers
templates/index.html Upload form
```

## Getting started

```bash
git clone https://github.com/khushwant04/ResNet-50-Classification.git
cd ResNet-50-Classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5000
```

`HOST`, `PORT`, `MODEL_PATH` and `FLASK_DEBUG=1` can be set through environment variables.

### Docker

```bash
docker build -t animal-classifier .
docker run --rm -p 5000:5000 animal-classifier
```

The image installs CPU-only PyTorch and serves the app with gunicorn.

## API

```bash
curl -F "file=@cat.jpg" http://127.0.0.1:5000/classify
# {"result": "Cat"}
```

On failure the response is `{"error": "..."}`.

## Training it yourself

Put images in `src/data/animal_data/<ClassName>/*.jpg` (one folder per class) and run `main.ipynb`. The dataset is not in this repository.

Note: in the notebook, `test_loader` is built from `train_data`. The "Test" accuracy it prints is therefore measured on the training split and does not show how well the model generalizes.

## License

MIT. See [LICENSE](LICENSE).
