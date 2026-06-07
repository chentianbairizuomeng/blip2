# Mini-BLIP2 Code

Put the Flickr8k Kaggle files under `data/` before training. The loader supports common layouts such as:

```text
data/
  Images/
  captions.txt
```

or:

```text
data/
  Flickr8k_Dataset/
  Flickr8k.token.txt
```

Check the dataset with:

```powershell
python -m code.dataset --data-root data --limit-images 200
```
