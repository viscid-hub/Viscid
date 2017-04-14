"""Sphinx plugin to generate matplotlib style examples

Modified from the seaborn project from seaborn/doc/sphinxext/plot_generator.py

Copyright (c) 2012-2013, Michael L. Waskom
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the {organization} nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
from __future__ import division
import os
import os.path as op
from pprint import pformat
import textwrap
import re

import matplotlib
matplotlib.use('Agg')
from matplotlib import image
import matplotlib.pyplot as plt

import viscid  # pylint: disable=unused-import

RST_TEMPLATE = """

{header}

.. image:: {img_file}
   :scale: 75 %

Style Sheet
~~~~~~~~~~~

Note that this is not the same syntax as the original style sheet.

.. code-block:: text
{style_spec}

Example Source Code
~~~~~~~~~~~~~~~~~~~

.. code-block:: python
{code}
"""


INDEX_TEMPLATE = """

Matplotlib Style Gallery
========================

If you have Matplotlib 1.5 or higher, you really should use a style sheet
for your plots. They are very easy to use as
`explained here <http://matplotlib.org/users/style_sheets.html>`_. The gallery
below summarizes all the styles provided by the recent versions of
Matplotlib and Viscid. A default set of style sheets can be loaded
using a viscidrc file as explained in :doc:`custom_behavior`.

.. raw:: html

    <style type="text/css">
    .autogen.figure {{
        position: relative;
        float: left;
        margin: 10px;
        width: 350px;
        height: 230px;
    }}

    .autogen.figure img {{
        position: absolute;
        display: inline;
        left: 0;
        width: 330px;
        height: 210px;
        border: 2px solid #2C3E52;
        opacity: 1.0;
        filter:alpha(opacity=100); /* For IE8 and earlier */
    }}

    .autogen.figure:hover img {{
        -webkit-filter: blur(3px);
        -moz-filter: blur(3px);
        -o-filter: blur(3px);
        -ms-filter: blur(3px);
        filter: blur(3px);
        opacity: 1.0;
        filter:alpha(opacity=100); /* For IE8 and earlier */
    }}

    .autogen.figure span {{
        position: absolute;
        display: inline;
        left: 0;
        width: 330px;
        height: 210px;
        background: #000;
        color: #fff;
        visibility: hidden;
        opacity: 0;
        z-index: 100;
    }}

    .autogen.figure p {{
        position: absolute;
        left: 0px;
        top: 45%;
        width: 240px;
        margin-left: 45px;
        margin-right: 45px;
        font-size: 105%;
        border-radius: 6px;
        background-color: hsla(0, 0%, 8%, 0.6);
        color: hsl(0, 0%, 96%);
    }}

    .autogen.figure:hover span {{
        visibility: visible;
        opacity: .4;
    }}

    .autogen.caption {{
        position: absolue;
        width: 330px;
        top: 210px;
        text-align: center !important;
    }}

    .autogen a {{
    }}

    span.auto_title {{
        position: relative;
        top: -10px;
        color: #2C3E52;
        font-size: 100%;
        text-align: center !important;
    }}
    </style>


{toctree}

{contents}

.. raw:: html

    <div style="clear: both"></div>
"""

GALERY_ENTRY_TEMPLATE = """\
.. raw:: html

    <div class='autogen figure align-center'>
    <a class='autogen' href=./{html_fname}>
    <img class='autogen' src=_static/{thumb_fname}>
    <span class='autogen figure-label'></span>
    <p class='autogen'>{style_name}</p>
    </a>
    </div>

"""

make_single_figure = r"""
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import viscid
from viscid.plot import vpyplot as vlt

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

fld = viscid.empty((np.linspace(1, 5, 64), np.linspace(1, 5, 64)),
                   name="F", pretty_name="Generic Field")
X, Y = fld.get_crds(shaped=True)
fld[:, :] = 1e-4 * (np.sin(X)**10 + np.cos(10 + X * Y) * np.cos(X))

with plt.style.context(("{style}",)):
    fig = plt.figure(figsize=(11, 7))

    ax = plt.subplot2grid((2, 9), (0, 0), rowspan=2)
    pal = vlt.get_current_colorcycle()
    size = 1
    n = len(pal)
    ax.imshow(np.arange(n).reshape(n, 1), cmap=ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylabel("Color Cycle")

    plt.subplot2grid((2, 9), (0, 1), rowspan=2, colspan=4)
    x = np.linspace(0, 2 * np.pi)
    for phase in np.linspace(0, np.pi / 4, n):
        plt.plot(x, (1 + np.sqrt(phase)) * np.sin(x - phase),
                     label=r"$\phi = {{0:.2g}}$".format(phase))
    plt.legend(loc=0)

    plt.subplot2grid((2, 9), (0, 5), colspan=4)
    vlt.plot(fld)
    plt.title("Sequential")

    plt.subplot2grid((2, 9), (1, 5), colspan=4)
    vlt.plot(fld, lin=0)
    plt.title("Symmetric")

    vlt.auto_adjust_subplots(subplot_params=dict(top=0.93, bottom=0.1))
    txt = ("Matplotlib Version: {{0}}\nViscid Version: {{1}}"
           "".format(matplotlib.__version__, viscid.__version__))
    fig.text(0.05, 0.01, txt, color='grey', size='small')
"""

save_single_figure = r"""
    import os
    plt.savefig("{img_fname}")
    plt.close()
"""


def create_thumbnail(infile, thumbfile,
                     width=300, height=300,
                     cx=0.5, cy=0.5, border=4):
    im = image.imread(infile)
    rows, cols = im.shape[:2]
    x0 = int(cx * cols - .5 * width)
    y0 = int(cy * rows - .5 * height)
    xslice = slice(x0, x0 + width)
    yslice = slice(y0, y0 + height)
    thumb = im[yslice, xslice]
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    ax = fig.add_axes([0, 0, 1, 1], aspect='auto',
                      frameon=False, xticks=[], yticks=[])
    ax.imshow(thumb, aspect='auto', resample=True,
              interpolation='bilinear')
    fig.savefig(thumbfile, dpi=dpi)
    plt.close()
    return None

def create_scaled_thumbnail(infile, thumbfile, scale=0.25, xl=0.0, yl=0.0,
                            xh=1.0, yh=1.0, border=0, border_value=0):
    img = image.imread(infile)
    ny, nx = img.shape[:2]

    xslice = slice(int(xl * nx), int(xh * nx))
    yslice = slice(int(yl * ny), int(yh * ny))
    thumb = img[yslice, xslice]
    if border:
        thumb[:border, :, :3] = border_value
        thumb[-border - 2:, :, :3] = border_value
        thumb[:, :border, :3] = border_value
        thumb[:, -border - 2:, :3] = border_value

    dpi = 100
    width, height = scale * nx, scale * ny
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], aspect='auto',
                      frameon=False, xticks=[], yticks=[])
    ax.imshow(thumb, aspect='auto', resample=True,
              interpolation='kaiser')
    fig.savefig(thumbfile, dpi=dpi)
    plt.close()
    return None

def indent(s, N=4):
    """indent a string"""
    return s.replace('\n', '\n' + N * ' ')

def main(app):
    target_dir = 'styles'
    image_dir = op.join('styles', '_images')
    thumb_dir = op.join('styles', '_thumbs')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    if not op.exists(image_dir):
        os.makedirs(image_dir)

    if not op.exists(thumb_dir):
        os.makedirs(thumb_dir)

    toctree = ("\n\n"
               ".. toctree::\n"
               "   :hidden:\n\n")
    contents = "\n\n"

    for style in list(sorted(matplotlib.style.available))[:]:
        img_fname = "{0}.png".format(style)
        thumb_fname = "{0}_thumb.png".format(style)
        html_fname = op.join('styles', "{0}.html".format(style))

        img_savename = op.join(image_dir, img_fname)
        thumb_savename = op.join(thumb_dir, thumb_fname)

        # eval's input is trusted, it's the string you see above
        cmd = make_single_figure.format(style=style)
        cmd_and_save = cmd + save_single_figure.format(img_fname=img_savename)
        eval(compile(cmd_and_save, '<string>', 'exec'), {}, {})

        spec = dict(matplotlib.style.library[style])
        style_spec = "\n"
        for key in sorted(spec.keys()):
            txt = pformat(spec[key])
            txt = re.sub(r"u'([^']*)'", r"\1", txt)
            lines = textwrap.wrap("{0}: {1}\n".format(key, txt), width=70)
            for i in range(1, len(lines)):
                lines[i] = " " * (len(key) + 2) + lines[i]
            style_spec += "{0}\n".format("\n".join(lines))

        create_scaled_thumbnail(img_savename, thumb_savename, scale=0.3)

        header = "Style: {0}".format(style)
        header = "{0}\n{1}".format(header, '=' * len(header))
        output = RST_TEMPLATE.format(header=header,
                                     img_file=op.join("_images", img_fname),
                                     style_spec=indent(style_spec, N=3),
                                     code=indent(cmd, N=3))
        with open(op.join(target_dir, "{0}.rst".format(style)), 'w') as f:
            f.write(output)

        toctree += "   {0}/{1}\n".format(target_dir, style)
        contents += GALERY_ENTRY_TEMPLATE.format(html_fname=html_fname,
                                                 thumb_fname=thumb_fname,
                                                 style_name=style)

    # write index file
    with open('mpl_style_gallery.rst', 'w') as index:
        index.write(INDEX_TEMPLATE.format(sphinx_tag="style_examples",
                                          toctree=toctree,
                                          contents=contents))

def setup(app):
    app.connect('builder-inited', main)
