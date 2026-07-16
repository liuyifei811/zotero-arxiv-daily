from .protocol import Paper
import math
from datetime import datetime


def get_stars(score: float, max_stars: int = 5) -> str:
    s = round(max(1, min(max_stars, score)))
    return '★' * s + '☆' * (max_stars - s)


def get_relevance_stars(score: float) -> str:
    if score is None:
        return '☆☆☆☆☆'
    s = min(5, max(1, round(score / 20)))
    return '⭐' * s


def get_relevance_pct(score: float) -> str:
    if score is None:
        return '--'
    return f"{min(99, max(50, round(score)))}%"


def format_date(d: datetime) -> str:
    if d is None:
        return '未知'
    return d.strftime('%Y年%m月%d日')


def render_email(papers: list[Paper]) -> str:
    if len(papers) == 0:
        return """
        <html><body style="font-family: sans-serif; padding: 20px;">
        <h2>今日无事，休息一下！</h2>
        <p>今天 arXiv 没有推送与你兴趣相关的新论文。</p>
        </body></html>
        """

    html_parts = []
    html_parts.append("""
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body { font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif; background: #f5f5f5; padding: 20px; color: #333; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 24px; }
        .header h1 { margin: 0; font-size: 24px; }
        .header p { margin: 8px 0 0; opacity: 0.85; font-size: 14px; }
        .paper { background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .paper-index { display: inline-block; background: #667eea; color: white; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: bold; margin-bottom: 12px; }
        .paper-meta { font-size: 12px; color: #999; margin-bottom: 8px; }
        .paper-title-cn { font-size: 18px; font-weight: bold; color: #1a1a2e; margin-bottom: 4px; }
        .paper-title-en { font-size: 14px; color: #888; margin-bottom: 16px; font-style: italic; }
        .section { margin-bottom: 10px; }
        .section-label { display: inline-block; font-weight: bold; color: #555; min-width: 90px; font-size: 14px; }
        .section-value { color: #333; font-size: 14px; }
        .relevance { font-size: 15px; }
        .datasets { display: inline-flex; gap: 6px; flex-wrap: wrap; }
        .dataset-tag { background: #e8f4fd; color: #1976d2; padding: 2px 10px; border-radius: 12px; font-size: 12px; }
        .code-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }
        .code-yes { background: #e8f5e9; color: #2e7d32; }
        .code-no { background: #fff3e0; color: #e65100; }
        .abstract-cn { background: #fafafa; padding: 12px 16px; border-radius: 8px; border-left: 3px solid #667eea; margin: 8px 0; font-size: 14px; line-height: 1.7; color: #444; }
        .innovation { margin: 8px 0; padding-left: 0; list-style: none; }
        .innovation li { padding: 4px 0; font-size: 14px; color: #333; }
        .innovation li::before { content: '💡 '; }
        .links { margin-top: 12px; }
        .links a { display: inline-block; padding: 6px 16px; border-radius: 20px; text-decoration: none; font-size: 13px; margin-right: 8px; font-weight: bold; }
        .link-pdf { background: #667eea; color: white; }
        .link-arxiv { background: #f0f0f0; color: #555; }
        .link-code { background: #2e7d32; color: white; }
        .footer { text-align: center; color: #aaa; font-size: 12px; margin-top: 30px; padding: 20px; }
    </style>
    </head>
    <body>
    <div class="header">
        <h1>📄 arXiv 每日论文推荐</h1>
        <p>基于你的 Zotero 文献库智能筛选 | 共 {total} 篇</p>
    </div>
    """.replace('{total}', str(len(papers))))

    for idx, p in enumerate(papers, 1):
        relevance_stars = get_relevance_stars(p.score) if p.score else '⭐⭐⭐'
        relevance_pct = get_relevance_pct(p.score)

        worth = p.worth_reading if p.worth_reading else 3
        worth_stars = get_stars(worth)

        pub_date_str = format_date(p.pub_date)

        has_code = p.has_code or (p.code_url is not None)
        code_html = ""
        if has_code and p.code_url:
            code_html = f'<span class="code-badge code-yes">✅ 有代码</span> <a class="link-code" href="{p.code_url}">GitHub</a>'
        elif has_code:
            code_html = '<span class="code-badge code-yes">✅ 有代码</span>'
        else:
            code_html = '<span class="code-badge code-no">❌ 未提供代码</span>'

        datasets_html = ""
        if p.datasets:
            tags = ''.join(f'<span class="dataset-tag">{d}</span>' for d in p.datasets[:8])
            datasets_html = f'<div class="datasets">{tags}</div>'

        author_list = [a for a in p.authors]
        if len(author_list) <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])

        paper_html = f"""
    <div class="paper">
        <div class="paper-index">第 {idx} 篇 · 推荐指数 {worth_stars}</div>
        <div class="paper-meta">📅 {pub_date_str} · 👤 {authors}</div>
        <div class="paper-title-cn">{p.chinese_title or p.title}</div>
        <div class="paper-title-en">{p.title}</div>

        <div class="section">
            <span class="section-label">与你课题相关性</span>
            <span class="relevance">{relevance_stars}（{relevance_pct}）</span>
        </div>

        <div class="section">
            <span class="section-label">为什么推荐</span>
            <span class="section-value">{p.recommendation_reason or '与你的研究兴趣匹配'}</span>
        </div>

        <div class="abstract-cn">
            <strong>中文摘要：</strong>{p.chinese_abstract or p.abstract[:300]}
        </div>

        <div class="section">
            <span class="section-label">创新点</span>
        </div>
        <div class="innovation">{p.innovation or '暂未分析'}</div>

        <div class="section">
            <span class="section-label">数据集</span>
            {datasets_html or '<span class="section-value">未提及</span>'}
        </div>

        <div class="section">
            <span class="section-label">是否值得精读</span>
            <span class="section-value">{worth_stars}</span>
        </div>

        <div class="links">
            <a class="link-pdf" href="{p.pdf_url or p.url}">📄 PDF</a>
            <a class="link-arxiv" href="{p.url}">arXiv</a>
            {code_html}
        </div>
    </div>"""
        html_parts.append(paper_html)

    html_parts.append("""
    <div class="footer">
        由 <strong>zotero-arxiv-daily</strong> 自动生成 | 如需退订，请在 GitHub Action 设置中移除邮箱
    </div>
    </body></html>""")

    return '\n'.join(html_parts)
