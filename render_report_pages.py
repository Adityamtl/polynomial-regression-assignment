import pymupdf

doc = pymupdf.open("BT2024188_Report.pdf")
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    pix.save(f"report_figures/report_page_{i+1}.png")
print("Rendered 4 report pages to PNG!")
