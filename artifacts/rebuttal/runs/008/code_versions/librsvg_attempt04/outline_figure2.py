"""Recover the submitted vector figure and remove font dependencies."""
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
REPO=RUN.parents[3]
sys.path.insert(0,'/tmp/daso_rebuttal_vector_tools')


def main():
    from pypdf import PdfReader,PdfWriter
    import cairocffi as cairo
    import numpy as np
    from PIL import Image
    original=REPO/'artifacts/rebuttal/reference/wu.pdf';reader=PdfReader(original)
    page=reader.pages[2];crop=(74,516,538,746)
    page.mediabox.lower_left=crop[:2];page.mediabox.upper_right=crop[2:]
    page.cropbox.lower_left=crop[:2];page.cropbox.upper_right=crop[2:]
    writer=PdfWriter();writer.add_page(page)
    extracted=RUN/'results/figure2_original_crop.pdf'
    with extracted.open('wb') as f:writer.write(f)
    svg=RUN/'results/figure2_outlined.svg'
    subprocess.run(['pdftocairo','-svg',str(extracted),str(svg)],check=True)
    root=ET.parse(svg).getroot();texts=[x for x in root.iter() if x.tag.rsplit('}',1)[-1] in ['text','tspan']]
    assert not texts,'SVG must contain glyph paths, not live text'
    # librsvg implements Poppler's luminance masks; CairoSVG's limited filter
    # support renders those image-backed panels black (retained diagnostics).
    cairo.ffi.cdef('typedef struct _RsvgHandle RsvgHandle; RsvgHandle *rsvg_handle_new_from_file(const char *, void **); int rsvg_handle_render_cairo(RsvgHandle *, cairo_t *); void rsvg_handle_set_dpi(RsvgHandle *, double); void g_object_unref(void *);')
    rsvg=cairo.ffi.dlopen('librsvg-2.so.2')
    handle=rsvg.rsvg_handle_new_from_file(str(svg).encode(),cairo.ffi.NULL);assert handle!=cairo.ffi.NULL
    rsvg.rsvg_handle_set_dpi(handle,72.)
    # SVG filters are rasterized by librsvg at the current surface resolution.
    # Render at 4x physical size, then scale the PDF coordinates back; this keeps
    # filtered original image panels sharp at the 288dpi inspection resolution.
    intermediate=RUN/'results/figure2_librsvg_4x.pdf';surface=cairo.PDFSurface(str(intermediate),464*4,230*4);context=cairo.Context(surface);context.scale(4,4)
    assert rsvg.rsvg_handle_render_cairo(handle,context._pointer)
    surface.finish();rsvg.g_object_unref(handle)
    outlined=PdfReader(intermediate).pages[0];outlined.scale_by(.25)
    target=RUN/'results/figure2_outlined.pdf';writer=PdfWriter();writer.add_page(outlined)
    with target.open('wb') as f:writer.write(f)
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
    versions={n:importlib.metadata.version(n) for n in ['cairocffi','pypdf','Pillow']}
    report={'passed':True,'source':str(original),'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'page_one_based':3,'crop_box_pdf_points':crop,'live_text_elements':len(texts),
        'output_pdf_font_count':0,'render_pixels':list(a.shape),'mean_abs_pixel_difference_0_255':difference,
        'fraction_pixels_any_channel_difference_gt_32':changed,
        'pdffonts_output':fonts,'versions':versions,
        'scope':'Original submitted Figure2 extracted as vector artwork and glyphs outlined through Poppler SVG/librsvg/Cairo. Pixel comparison and visual inspection required. No diagram semantics or formulas changed. Original embedded-font PDF already rendered legibly here; this removes font/viewer substitution dependence, not proof of a universally reproduced reviewer-side failure.'}
    (RUN/'results/figure2_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (RUN/'README.md').write_text('# Run008 — 人工reviewer文档与图形补充\n\n'
        '原reference论文保持原样。Figure2在本机可以正确渲染，原字体已嵌入；另提供字形全部转为路径的矢量PDF/SVG，消除字体替换依赖。未改公式语义，未声称复现所有reviewer端显示问题。\n\n'
        '[无字体依赖PDF](results/figure2_outlined.pdf) · [SVG](results/figure2_outlined.svg) · [渲染图](results/figure2_outlined.png) · [审计](results/figure2_audit.json)。\n\n'
        '重现：独立安装cairocffi/pypdf到`/tmp/daso_rebuttal_vector_tools`，版本及pip日志归档；执行code/outline_figure2.py，使用系统librsvg/Cairo及pdftocairo/pdftoppm/pdffonts。CairoSVG对原luminance mask支持不足的失败样本保留在diagnostics。\n')
    print(json.dumps(report))


if __name__=='__main__':main()
