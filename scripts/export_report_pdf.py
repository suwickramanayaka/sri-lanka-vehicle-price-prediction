"""Render the Markdown report as PDF. Requires reportlab (optional report dependency)."""
from pathlib import Path
import re,html,argparse
import reportlab
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='deliverables/vehicle-price-prediction-ngboost-report.pdf')
    args=parser.parse_args()
    source=ROOT/'reports/final_report.md';out=ROOT/args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    fonts=Path(reportlab.__file__).parent/'fonts'
    for name,file in [('Body','Vera.ttf'),('BodyBold','VeraBd.ttf'),('BodyItalic','VeraIt.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
    styles=getSampleStyleSheet()
    for name in ['Normal','BodyText','Heading1','Heading2','Heading3']:
        styles[name].fontName='Body';styles[name].textColor=colors.HexColor('#243441')
    styles['BodyText'].fontSize=9.3;styles['BodyText'].leading=14;styles['BodyText'].spaceAfter=7
    for name,size in [('Heading1',18),('Heading2',14),('Heading3',11)]:
        style=styles[name];style.keepWithNext=True;style.fontName='BodyBold';style.fontSize=size;style.leading=size+5;style.spaceBefore=14;style.spaceAfter=9
    styles.add(ParagraphStyle(name='Cell',fontName='Body',fontSize=7.4,leading=10,textColor=colors.HexColor('#243441')))
    styles.add(ParagraphStyle(name='Caption',fontName='BodyItalic',fontSize=8,leading=11,textColor=colors.HexColor('#526674'),spaceAfter=10))
    styles.add(ParagraphStyle(name='Cover',fontName='BodyBold',fontSize=29,leading=37,textColor=colors.HexColor('#153b48'),spaceAfter=24))
    W,H=A4;width=W-88
    def inline(text):
        for old,new in [('–','-'),('—','-'),('−','-'),('→','to'),('²','2'),('’',"'"),('“','"'),('”','"')]:text=text.replace(old,new)
        text=html.escape(text)
        text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2" color="#087f8c">\1</a>',text)
        text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
        return re.sub(r'`([^`]+)`',r'<font size="8">\1</font>',text)
    story=[Spacer(1,80),Paragraph('TECHNICAL REPORT',styles['Heading3']),Spacer(1,16),Paragraph('Sri Lankan Vehicle<br/>Listing Price Prediction',styles['Cover']),Paragraph('NGBoost probabilistic regression<br/>and comparative model evaluation',styles['Heading2']),Spacer(1,35),Paragraph('Sithum Wickramanayaka',styles['Heading2']),Paragraph('University of Moratuwa<br/>MSc in Artificial Intelligence<br/>IT5514 - Applied Machine Learning<br/>Student ID: 269214N | Batch 19',styles['BodyText']),Paragraph('28 October 2026',styles['BodyText']),Spacer(1,30),Paragraph('Historical vehicle advertisements | December 2024 - February 2025',styles['Caption']),PageBreak()]
    lines=source.read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line or line.startswith('# ') or line=='## Technical Report' or line.startswith('**Author:**'):i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
                i+=1
            n=len(rows[0]);cols=[width/n]*n
            if n>=5:cols=[width*.22]+[width*.78/(n-1)]*(n-1)
            table=Table([[Paragraph(inline(c) if r else '<b>'+inline(c)+'</b>',styles['Cell']) for c in row] for r,row in enumerate(rows)],colWidths=cols,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5f1f2')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f8')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
            if len(rows)<=13:
                prefix=[]
                if story and isinstance(story[-1],Paragraph) and story[-1].style.name.startswith('Heading'):prefix=[story.pop()]
                story.append(KeepTogether(prefix+[table]))
            else:story.append(table)
            story.append(Spacer(1,12));continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));title=line[level:].strip()
            if title.startswith('Appendix A') or (level==3 and 'strengths and limitations' in title):story.append(PageBreak())
            story.append(Paragraph(inline(title),styles['Heading2' if level==2 else 'Heading3']));i+=1;continue
        match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if match:
            img=Image(str(source.parent/match[2]));ratio=min(width/img.imageWidth,250/img.imageHeight)
            img.drawWidth=img.imageWidth*ratio;img.drawHeight=img.imageHeight*ratio
            story.append(KeepTogether([img,Spacer(1,5),Paragraph(inline(match[1]),styles['Caption'])]));i+=1;continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','- ')) and not re.match(r'^\d+\. ',lines[i]):para.append(lines[i].strip());i+=1
        text=' '.join(para)
        story.append(Paragraph(inline(text),styles['BodyText']))
    def footer(canvas,doc):
        if doc.page==1:return
        canvas.setStrokeColor(colors.HexColor('#d8e4e8'));canvas.line(44,40,W-44,40)
        canvas.setFont('Body',7);canvas.setFillColor(colors.HexColor('#526674'));canvas.drawString(44,27,'VEHICLE LISTING PRICE PREDICTION');canvas.drawRightString(W-44,27,str(doc.page))
    SimpleDocTemplate(str(out),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=40,bottomMargin=55,title='Sri Lankan Vehicle Listing Price Prediction - NGBoost study',author='Sithum Wickramanayaka').build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)

if __name__=='__main__':main()
