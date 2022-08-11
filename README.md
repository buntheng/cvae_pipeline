# Python API 

# Installation & Requirements

- Pull this package
``` shell
> git clone https://github.com/buntheng/cvae_pipeline.git 
```

- Third party packages
    - SimpleITK
    - vtk
    - pyvista
    - pyacvd
    - tensorflow
    - tensorflow-addons
    - cython # See Note bellow.
    - pyezzi # See Note bellow.
    - scikit-learn

```
pip install -r requirements.txt
```

- **Known bug**: `cython` and `pyezzi`
Please make sure that `pyezzi` is compiled correctly, which required `cython`.
In test machine using `conda`, we had to first install `cython` with `conda install cython`, before installing pyezzi with `pip install pyezzi`.

## Complete Prediction Pipeline
```python
python main.py
```

Main API functions can be found in `automate.py`


