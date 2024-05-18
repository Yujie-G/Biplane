# [Biplane](https://wangningbei.github.io/2023/BIPLANEBTF.html)

## dependencies

```bash
conda create -n biplane --file requirements.txt
```

Mitsuba Renderer Part(Optional, if you want to use Biplane mat model to render new scene)
[Mtisuba 0.6](https://github.com/Yujie-G/mitsuba0.6/tree/Biplane)


## Usage

After all setup in config.py, you can run the following command to start the training process.
```bash
python main.py
```

### Render new mitsuba scene

set the model path in xml file:

```xml
<bsdf type="decoder_gaussian" id="dummy">
    <integer name="index" value="0"/>
    <string name="checkpointPath" value="/path/to/your/model.pth"/>
    <float name="sigma" value="0.0"/>
    <float name="amp" value="0.0"/>
</bsdf>
```


After sepecify the model path in xml file, you can render the scene by running the following command.
```bash
python render_script/render_xml_scene.py /path/to/your/xml_file_path.xml
```




