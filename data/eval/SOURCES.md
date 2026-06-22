# Evaluation Corpus Sources

This evaluation corpus (`data/eval/raw/`) is built from **real, externally authored public documents**
so retrieval and generation metrics are not trivially overfit to project-authored text. Each entry below
is a short excerpt used for evaluation, with its source and licensing/terms.

| Document | Category | Source | License / Terms |
| --- | --- | --- | --- |
| `gdpr_art5_principles.md` | legal | https://gdpr-info.eu/art-5-gdpr/ | Regulation (EU) 2016/679 (GDPR); EU legal text, reuse permitted with attribution |
| `gdpr_art17_erasure.md` | legal | https://gdpr-info.eu/art-17-gdpr/ | Regulation (EU) 2016/679 (GDPR); EU legal text, reuse permitted with attribution |
| `gdpr_art33_breach.md` | legal | https://gdpr-info.eu/art-33-gdpr/ | Regulation (EU) 2016/679 (GDPR); EU legal text, reuse permitted with attribution |
| `eu_ai_act_risk_tiers.md` | legal | https://artificialintelligenceact.eu/high-level-summary/ | Regulation (EU) 2024/1689; summary of EU legal text, reuse permitted with attribution |
| `transformer_attention.md` | academic | https://arxiv.org/abs/1706.03762 | Vaswani et al., 2017, arXiv:1706.03762; abstract excerpt cited for research use |
| `bert.md` | academic | https://arxiv.org/abs/1810.04805 | Devlin et al., 2018, arXiv:1810.04805; abstract excerpt cited for research use |
| `resnet.md` | academic | https://arxiv.org/abs/1512.03385 | He et al., 2015, arXiv:1512.03385; abstract excerpt cited for research use |
| `adam_optimizer.md` | academic | https://arxiv.org/abs/1412.6980 | Kingma and Ba, 2014, arXiv:1412.6980; abstract excerpt cited for research use |
| `rfc2119_keywords.md` | technical | https://www.rfc-editor.org/rfc/rfc2119.txt | IETF RFC 2119 (Bradner, 1997); reproducible per IETF Trust legal provisions |
| `rfc8446_tls13.md` | technical | https://www.rfc-editor.org/rfc/rfc8446.txt | IETF RFC 8446 (Rescorla, 2018); reproducible per IETF Trust legal provisions |
| `rfc9110_http_methods.md` | technical | https://www.rfc-editor.org/rfc/rfc9110.txt | IETF RFC 9110 (Fielding et al., 2022); reproducible per IETF Trust legal provisions |
| `rfc1035_dns.md` | technical | https://www.rfc-editor.org/rfc/rfc1035.txt | IETF RFC 1035 (Mockapetris, 1987); reproducible per IETF Trust legal provisions |
| `nist_csf_functions.md` | security | https://www.nist.gov/cyberframework | NIST CSWP 29, The NIST Cybersecurity Framework 2.0; U.S. Government work, public domain |
| `nist_ai_rmf_functions.md` | security | https://www.nist.gov/itl/ai-risk-management-framework | NIST AI 100-1, Artificial Intelligence Risk Management Framework; U.S. Government work, public domain |
| `nist_zero_trust.md` | security | https://csrc.nist.gov/glossary/term/zero_trust | NIST SP 800-207, Zero Trust Architecture; U.S. Government work, public domain |
| `sec_form_10k.md` | finance | https://www.investor.gov/introduction-investing/investing-basics/glossary/form-10-k | SEC / Investor.gov; U.S. Government work, public domain |
| `sec_diversification.md` | finance | https://www.investor.gov/introduction-investing/investing-basics/glossary/diversification | SEC / Investor.gov; U.S. Government work, public domain |
| `sec_mutual_funds.md` | finance | https://www.investor.gov/introduction-investing/investing-basics/glossary/mutual-funds | SEC / Investor.gov; U.S. Government work, public domain |

Notes:
- GDPR / EU AI Act excerpts are EU legal texts; reuse is permitted with attribution.
- IETF RFC excerpts are reproducible under the IETF Trust legal provisions.
- NIST and SEC/Investor.gov materials are U.S. Government works in the public domain.
- arXiv abstracts are short excerpts cited for research/evaluation use with attribution to the authors.
- Corpus size: 18 documents across 5 categories.

