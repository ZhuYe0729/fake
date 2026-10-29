"""Recover the submitted vector figure and remove font dependencies."""
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
REPO=RUN.parents[3]
sys.path.insert(0,'/tmp/daso_rebuttal_vector_tools')


def main():
    from pypdf import PdfReader,PdfWriter
    import numpy as np
    from PIL import Image
    original=REPO/'artifacts/rebuttal/reference/wu.pdf';reader=PdfReader(original)
    page=reader.pages[2];crop=(74,512,538,756)
    page.mediabox.lower_left=crop[:2];page.mediabox.upper_right=crop[2:]
    page.cropbox.lower_left=crop[:2];page.cropbox.upper_right=crop[2:]
    writer=PdfWriter();writer.add_page(page)
    extracted=RUN/'results/figure2_original_crop.pdf'
    with extracted.open('wb') as f:writer.write(f)
    svg=RUN/'results/figure2_outlined.svg'
    subprocess.run(['pdftocairo','-svg',str(extracted),str(svg)],check=True)
    root=ET.parse(svg).getroot();texts=[x for x in root.iter() if x.tag.rsplit('}',1)[-1] in ['text','tspan']]
    assert not texts,'SVG must contain glyph paths, not live text'
    # Native PDF outlining preserves the original image masks and resolution.
    # Earlier SVG conversion failures are archived, never used as final output.
    gsroot=Path('/tmp/daso_rebuttal_ghostscript');gs=gsroot/'usr/bin/gs'
    env={**os.environ,'LD_LIBRARY_PATH':str(gsroot/'usr/lib/x86_64-linux-gnu'),
        'GS_LIB':':'.join(str(gsroot/p) for p in ['usr/share/ghostscript/10.02.1/Resource/Init','usr/share/ghostscript/10.02.1/Resource/Font','usr/share/ghostscript/10.02.1/lib'])}
    target=RUN/'results/figure2_outlined.pdf'
    command=[str(gs),'-q','-dNOPAUSE','-dBATCH','-dSAFER','-sDEVICE=pdfwrite','-dNoOutputFonts','-dAutoRotatePages=/None',
        '-dDownsampleColorImages=false','-dDownsampleGrayImages=false','-dDownsampleMonoImages=false','-sOutputFile='+str(target),str(extracted)]
    subprocess.run(command,check=True,env=env)
    for source,name in [(extracted,'figure2_original_crop'),(target,'figure2_outlined')]:
        subprocess.run(['pdftoppm','-r','288','-singlefile','-png',str(source),str(RUN/'results'/name)],check=True)
    a=np.array(Image.open(RUN/'results/figure2_original_crop.png').convert('RGB')).astype(float)
    b=np.array(Image.open(RUN/'results/figure2_outlined.png').convert('RGB')).astype(float)
    assert a.shape==b.shape,(a.shape,b.shape)
    fonts=subprocess.check_output(['pdffonts',str(target)],text=True)
    # pdffonts emits two heading lines and no rows for path-only output.
    assert len(fonts.strip().splitlines())==2,fonts
    difference=float(abs(a-b).mean());changed=float((abs(a-b).max(2)>32).mean())
    assert difference<5 and changed<.05,(difference,changed)
    versions={n:importlib.metadata.version(n) for n in ['pypdf','Pillow']}
    versions['ghostscript']=subprocess.check_output([str(gs),'--version'],env=env,text=True).strip()
    report={'passed':True,'source':str(original),'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'page_one_based':3,'crop_box_pdf_points':crop,'live_text_elements':len(texts),
        'output_pdf_font_count':0,'render_pixels':list(a.shape),'mean_abs_pixel_difference_0_255':difference,
        'fraction_pixels_any_channel_difference_gt_32':changed,
        'pdffonts_output':fonts,'versions':versions,
        'command':command,'scope':'Original submitted Figure2 extracted and outlined through native Ghostscript pdfwrite, without image downsampling. Poppler also exports a path-based SVG. Pixel comparison and visual inspection required. No diagram semantics or formulas changed. Original embedded-font PDF already rendered legibly here; this removes font/viewer substitution dependence, not proof of a universally reproduced reviewer-side failure.'}
    (RUN/'results/figure2_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (RUN/'README.md').write_text('# Run008 — 人工reviewer文档与图形补充\n\n'
        '原reference论文保持原样。Figure2在本机可以正确渲染，原字体已嵌入；另提供字形全部转为路径的矢量PDF/SVG，消除字体替换依赖。未改公式语义，未声称复现所有reviewer端显示问题。\n\n'
        '[无字体依赖PDF](results/figure2_outlined.pdf) · [SVG](results/figure2_outlined.svg) · [渲染图](results/figure2_outlined.png) · [审计](results/figure2_audit.json)。\n\n'
        '重现：pypdf工具装在`/tmp/daso_rebuttal_vector_tools`，Ghostscript Debian包解压在`/tmp/daso_rebuttal_ghostscript`，没有全局安装；执行code/outline_figure2.py，使用pdftocairo/pdftoppm/pdffonts。SVG转PDF的mask/低分辨率失败样本及脚本保留。\n')
    print(json.dumps(report))


if __name__=='__main__':
    try:main()
    except Exception as e:
        (RUN/'results/figure2_audit.json').write_text(json.dumps({'passed':False,'error':repr(e)},indent=2)+'\n')
        raise
