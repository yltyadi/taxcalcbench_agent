# China and Indonesia official source packs

Source preparation was checked on 2026-09-22 using the actual `taxcalcbench download` command, without model calls. Subsequent one-family generation checks are recorded in [HPC preparation checks](hpc-checks.md). These are bounded starting corpora, not complete national tax-law archives.

| Country | Config | Local corpus | Captured | Extraction |
|---|---|---|---:|---|
| China, Chinese (`zh`) | [`configs/cn.json`](../configs/cn.json) | `data/cn/sources` | 15/15 | 14 complete text captures; one partial PDF |
| Indonesia, Indonesian (`id`) | [`configs/id.json`](../configs/id.json) | `data/id/sources` | 13/13 | 12 complete text captures; one partial HTML document |

Both packs use exact HTTPS host/path allowlists and fixed document URLs, with no crawling (`link_rules: []`, `max_depth: 0`). Only national authorities are included. The configured 30-document acquisition ceiling is above the number of seeds; it does not truncate either pack. The `en/node/...` Indonesian URLs contain Indonesian legal text, not English translations.

## Reproduce

```bash
.venv/bin/taxcalcbench download --country configs/cn.json --source-dir data/cn/sources
.venv/bin/taxcalcbench download --country configs/id.json --source-dir data/id/sources
```

The usual `run --country ... --source-dir ... --output ...` command then uses the captured corpus. Keep each entire source folder with its output checkpoints when moving to another machine. Files and source hashes are local; no database or API key is needed for downloading.

## China

The approved publishers are the central government (`www.gov.cn`), Ministry of Finance (`www.mof.gov.cn`, `jgxt.mof.gov.cn`), and National Council for Social Security Fund (`www.ssf.gov.cn`). The latter hosts official national statutory PDFs. The initial Ministry of Finance VAT URL returned HTTP 502; the final config uses its working central Ministry of Finance legal-system mirror.

| Official document | Main contribution |
|---|---|
| [中华人民共和国企业所得税法（2007年通过）](https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/2007zcfb/200805/t20080519_26016.htm) | Income tax from 2008; enacted text |
| [中华人民共和国企业所得税法实施条例（2007年公布）](https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/2007zcfb/200805/t20080519_27853.htm) | Corporate-tax calculation rules from 2008 |
| [中华人民共和国个人所得税法（2011年修正）](https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/201107/t20110701_569188.htm) | Historical personal-income-tax text and brackets |
| [中华人民共和国个人所得税法（2018年修正）](https://www.ssf.gov.cn/portal/rootfile/P020180917424641370482.pdf) | Revised personal-income-tax text and brackets |
| [中华人民共和国个人所得税法实施条例（国务院令第707号）](https://www.gov.cn/zhengce/content/2018-12/22/content_5351177.htm) | Implementing rules effective 2019 |
| [个人所得税专项附加扣除暂行办法（国发〔2018〕41号）](https://www.gov.cn/zhengce/content/2018-12/22/content_5351181.htm) | 2019 additional deductions |
| [国务院关于提高个人所得税有关专项附加扣除标准的通知（国发〔2023〕13号）](https://www.gov.cn/zhengce/content/202308/content_6901206.htm) | 2023 changes to additional deductions |
| [中华人民共和国增值税暂行条例（国务院令第538号，2008年修订）](https://jgxt.mof.gov.cn/webfile/flfg/2019-08-05/76.html) | Historical VAT rules effective 2009 |
| [增值税暂行条例和营业税暂行条例实施细则（2011年修订，财政部令第65号）](https://www.gov.cn/gongbao/content/2012/content_2121706.htm) | Historical VAT/business-tax implementing provisions |
| [中华人民共和国增值税法（2024年通过，自2026年1月1日起施行）](https://www.ssf.gov.cn/portal/rootfiles/2025/01/17/1738811814919836-1738811814942199.pdf) | VAT law effective 2026 |
| [中华人民共和国印花税暂行条例（2011年修订）](https://www.gov.cn/gongbao/content/2011/content_1860821.htm) | Historical stamp-tax rules and rate table |
| [中华人民共和国印花税法（2021年通过，自2022年7月1日起施行）](https://www.ssf.gov.cn/portal/rootfiles/2022/07/29/1660719214618663-1660719214632636.pdf) | Modern stamp-tax provisions; rate-table PDF pages withheld |
| [中华人民共和国房产税暂行条例（2011年修订）](https://www.gov.cn/gongbao/content/2011/content_1860812.htm) | Property-tax provisions; some parameters require local rules |
| [中华人民共和国社会保险法（2018年修正）](https://www.gov.cn/guoqing/2021-10/29/content_5647616.htm) | National social-insurance framework |
| [国务院办公厅关于印发降低社会保险费率综合方案的通知（国办发〔2019〕13号）](https://www.gov.cn/zhengce/content/2019-04/04/content_5379629.htm) | National 2019 contribution-rate policy; local bases may still be required |

Limits: the 2021 stamp-law PDF has graphics on pages 6–7, including its rate schedule. Those pages are explicitly withheld, so this copy alone cannot support current stamp-rate calculations. Some 2018/2019 VAT rate amendments, personal-income-tax amendment commencement decisions, local property-tax parameters, and social-insurance bases/caps are not included. Publication and consolidation dates must not be treated as proof that every provision applied in an earlier year. The national sources provide material in OECD 1000, 2000, 4000 and 5000; they do not establish complete coverage of all six categories or every year.

## Indonesia

The approved publishers are the Ministry of Finance legal documentation service (`jdih.kemenkeu.go.id`) and national Directorate General of Taxes (`www.pajak.go.id`, `stats.pajak.go.id`). Several original Ministry PDFs were scans or contained unsupported graphics on every page; the final pack uses the same enactments in the national tax authority’s full HTML publication instead.

| Official document | Main contribution |
|---|---|
| [UU 18 TAHUN 2000 - Perubahan Kedua atas Undang-Undang Nomor 8 Tahun 1983 tentang Pajak Pertambahan Nilai Barang dan Jasa dan Pajak Penjualan atas Barang Mewah.](https://jdih.kemenkeu.go.id/api/download/fulltext/2000/18TAHUN2000UU.htm) | VAT amendment effective 2001 |
| [UU 42 TAHUN 2009 - Perubahan Ketiga atas Undang-Undang Nomor 8 Tahun 1983 tentang Pajak Pertambahan Nilai Barang dan Jasa dan Pajak Penjualan atas Barang Mewah.](https://www.pajak.go.id/en/node/35341) | VAT amendment effective 2010 |
| [PP 23 TAHUN 2018 - Pajak Penghasilan atas Penghasilan Dari Usaha yang Diterima atau Diperoleh Wajib Pajak yang Memiliki Peredaran Bruto Tertentu](https://www.pajak.go.id/en/node/58434) | SME turnover-tax regime from 2018 |
| [PP 55 TAHUN 2022 - Penyesuaian Pengaturan di Bidang Pajak Penghasilan](https://stats.pajak.go.id/id/peraturan/penyesuaian-pengaturan-di-bidang-pajak-penghasilan) | Income-tax implementation changes from 2022 |
| [UU 10 TAHUN 2020 - Bea Meterai](https://www.pajak.go.id/en/node/58274) | Stamp-duty statute effective 2021; one explanation withheld |
| [PP 24 TAHUN 2000 - Perubahan Tarif Bea Meterai dan Besarnya Batas Pengenaan Harga Nominal yang Dikenakan Bea Meterai.   Jakarta, 2000](https://jdih.kemenkeu.go.id/api/download/fulltext/2000/24TAHUN2000PP.Htm) | Historical stamp-duty rates |
| [UU 12 TAHUN 1994 - Perubahan atas Undang-Undang Nomor 12 Tahun 1985 tentang Pajak Bumi dan Bangunan.](https://jdih.kemenkeu.go.id/api/download/fulltext/1994/12TAHUN~1994UU.HTM) | National land/building-tax amendment; base law/supporting rules may be required |
| [101/PMK.010/2016 - Penyesuaian Besarnya Penghasilan Tidak Kena Pajak.](https://www.pajak.go.id/id/peraturan/penyesuaian-besarnya-penghasilan-tidak-kena-pajak-2) | Personal non-taxable allowance from 2016 |
| [PP 14 TAHUN 1993 - Penyelenggaraan Program Jaminan Sosial Tenaga Kerja.](https://jdih.kemenkeu.go.id/api/download/fulltext/1993/14TAHUN1993PP.htm) | Historical employee social-security rules |
| [UU 17 TAHUN 2000 — Perubahan Ketiga Undang-Undang Pajak Penghasilan](https://jdih.kemenkeu.go.id/api/download/fulltext/2000/17TAHUN2000UU.htm) | Historical income-tax amendment |
| [UU 36 TAHUN 2008 — Perubahan Keempat Undang-Undang Pajak Penghasilan](https://jdih.kemenkeu.go.id/api/download/fulltext/2008/36TAHUN2008UU.htm) | Income-tax amendment effective 2009 |
| [UU 7 TAHUN 2021 — Harmonisasi Peraturan Perpajakan](https://www.pajak.go.id/en/node/74838) | 2021 reform: income tax, VAT and other provisions with separate commencement dates |
| [PP 46 TAHUN 2013 — Pajak Penghasilan atas Penghasilan Dari Usaha dengan Peredaran Bruto Tertentu](https://jdih.kemenkeu.go.id/api/download/fulltext/2013/46TAHUN2013PP.htm) | Historical SME turnover-tax regime from 2013 |

Extraction excludes only two verified national-emblem image paths on the tax-authority hosts and six exact presidential-emblem paths in the older Ministry HTML files. The six older images have identical SHA-256 hashes and were visually checked; there is no blanket image removal. Substantive images remain gaps. In UU 10/2020, an image in the explanation of Pasal 11 is withheld; the enacted rate provisions are readable.

Limits: later social-security regimes, all historical personal allowance changes, the complete national land/building-tax base law and implementing parameters, and later VAT/SME amendments are not exhaustively covered. In particular, do not assume the 2022 SME regulation is unchanged throughout 2026. The corpus offers OECD 1000, historical 2000, 4000-related and 5000-related material; it does not establish all six categories as applicable or fully sourced.

The six OECD category settings remain planner preferences inherited from the shared defaults. Unsupported categories, missing supporting provisions and inapplicable years must be addressed by planning from the available text, not invented rates. A successful download verifies acquisition and extraction, not legal completeness or generated-question correctness.
