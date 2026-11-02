# Piracy, file sharing and DRM

A small test dataset of 30 English Wikinews articles (2004-12-21 to 2021-04-21),
hand-picked from keyword search results to be on topic, reasonably long, rich in quotes from named
people, and spread over time.

- Source: English Wikinews database dump of 2026-05-01 (https://en.wikinews.org)
- Words per article: min 202, median 553, max 6925; total 29784
- Articles per year: 2004: 1, 2005: 5, 2006: 5, 2007: 2, 2008: 1, 2009: 3, 2010: 4, 2011: 1, 2012: 4, 2016: 2, 2020: 1, 2021: 1

## Files
- `articles.jsonl`: one article per line: id, page_id, rev_id, title, date, url, license, categories,
  sources (cited by Wikinews), lead, text (plain text, without the Sources/Related sections),
  word_count, quote_count
- `articles.csv`: overview table
- `index.md`: readable list with links

## Licence and attribution
English Wikinews text is licensed by publication date (see each record's `license` field):
public domain before 25 September 2005; CC BY 2.5 until 15 December 2024; CC BY 4.0 afterwards.
In this subset: CC BY 2.5: 26, Public domain: 4.
When resharing, credit "Wikinews contributors" and link each article's `url`.
Wikinews was closed and made read-only on 4 May 2026.
