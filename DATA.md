# Datasets: sources, files and checksums

Neither dataset is included in this package. Both are public CDC data. Download them yourself and check them against the
checksums below. The stage runner refuses to run the BRFSS stages if the BRFSS file's SHA-256 differs.

## BRFSS 2024 (main analysis)

| Item | Value |
| --- | --- |
| Survey | CDC Behavioral Risk Factor Surveillance System, 2024 combined landline and cell-phone data (LLCP2024) |
| Official page | https://www.cdc.gov/brfss/annual_data/annual_2024.html |
| Official download | `LLCP2024XPT.zip` (SAS Transport): https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip; codebook `codebook24_llcp-v2-508.zip` |
| Copy used for the paper and the verified runs | Kaggle re-upload `rudritarahman/cdc-brfss-survey-data-2024`, file **`main_data_brfss_2024.XPT`** |
| Size | 1,093,874,240 bytes |
| SHA-256 | `7d53cf088f858e02bcc029a51cf0408cc68f40a0ebeb02a3c0f73058aab36b1c` |
| Contents as read by the code | 457,670 rows; 452,464 with a valid `_MICHD` label (42,338 positive) |
| Place it at | `data/brfss/main_data_brfss_2024.XPT`, or set the environment variable `HS_BRFSS_XPT` to its full path before generating `run_code/` and running the stages |

**Provenance notes (as found):**
- The paper's runs used the Kaggle copy. It has not been checked whether that file is byte-identical to the XPT inside CDC's `LLCP2024XPT.zip`.
- CDC's 2024 page states that the data set was modified to comply with executive orders. A file downloaded from CDC today may therefore differ from the copy used here.
- If you use the CDC file, check its SHA-256 against the value above first. If it differs, the pre-flight check will stop. You can then point `HS_BRFSS_XPT` to the Kaggle copy, or obtain the exact file from the author.
- The Kaggle dataset's licence terms were not recorded with the project. **[AUTHOR REVIEW: confirm the Kaggle licence before wider distribution.]**

## NHANES 2017–March 2020 pre-pandemic (external cohort)

Source: CDC/NCHS National Health and Nutrition Examination Survey, public-use files of the 2017–March 2020 pre-pandemic cycle (prefix `P_`). Each file downloads from
`https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/<FILE>.xpt`, and its documentation is at the same address with `.htm`.

Place all 18 files in `data/nhanes/` with exactly these names (lower-case `.xpt`):

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `P_DEMO.xpt` | 3,614,720 | `2e46c6c26bf77cd8989f64011ace12cbf42c0f3e03414eb59acc5328c8f87913` |
| `P_BMX.xpt` | 2,520,640 | `7038d0da3169420a4a3cbaec09ac586a701b38e1c7b925431b60e13ac24fed66` |
| `P_BPXO.xpt` | 1,039,840 | `5edbcb81e99fe8dad38deb8b642570278b200c525d88ffa29ae3ceb4dd38bde4` |
| `P_BPQ.xpt` | 899,520 | `88af3557dd185cee2533c94c5302b361e89a54eb50754704f35d4e8120868c02` |
| `P_DIQ.xpt` | 3,361,520 | `79153f799fc2171792771c4aa250029c62807f14915f1f81405b03233b1e5ae3` |
| `P_MCQ.xpt` | 7,562,560 | `1de8e4e184ed375fddc053dbb5b82db13ee55e4d4d0ae12da9e407537230125c` |
| `P_PAQ.xpt` | 1,321,440 | `d0e13cebf96181949e983ab58d14c58689e801e8711d366d2b5daab3c7b9ee65` |
| `P_SMQ.xpt` | 1,428,560 | `29b7f6c59ab570c042c866312f0eb4329009e85fadbda6f9427a8bf31ecb3a3f` |
| `P_ALB_CR.xpt` | 835,600 | `c493f6b74cc0608bb7083b92e074fc3070b9e614fd5e1e34d0ce943e0f6116bf` |
| `P_BIOPRO.xpt` | 3,420,640 | `f242cb9f9da5787f77d81ffb02822350e01cea4adf072d23bc96eb008182010c` |
| `P_CBC.xpt` | 2,427,760 | `8a910eacc0fad8d3b817bb01699c5d90761c60be3b2daaaf025396b58de96ef8` |
| `P_GHB.xpt` | 167,600 | `dac9e423f56041c2ec46486cb16be16d3347e0a81a575985882ffad5a60e1195` |
| `P_GLU.xpt` | 164,160 | `91b83c46fb4707c431838159135e93693a0572bce54074f5a890f0ee104732b3` |
| `P_HDL.xpt` | 294,000 | `04a344f00fe34b3dc4202ee9e47cf61e67d1f985d7c9ece228e25caa85c7fbd7` |
| `P_HSCRP.xpt` | 331,760 | `e6954d17afa70755c26004bda6d4f0b256a5f30b0620cb70fdd6f520147218e1` |
| `P_INS.xpt` | 205,040 | `00b2cebd564df7038451d297df1a1d11e8e69a9443eee16770c24cc2ce3471da` |
| `P_TCHOL.xpt` | 294,000 | `759cd9c408b5d1b91f6cfb5d5e67920a301f7478c0e90ce7adf4bbd277a45ac8` |
| `P_TRIGLY.xpt` | 409,360 | `d3eb2df386f24a559ae4db992f0ec9aaf3d2c0caec173aa1921e78100ee88b31` |

**Notes:**
- These checksums are those of the files used for the paper and the verified runs. CDC occasionally re-releases files, so a fresh download may differ.
- Stage 1 writes three derived CSVs next to the modules: `nhanes_2017_2020_merged_raw.csv`, `nhanes_cleaned_step1.csv` and `nhanes_final_with_target.csv`. Their checksums from the verified run are in `manifests/excluded_files.sha256`.
- NHANES public-use data are distributed under NCHS's data-use restrictions. Among other things, they prohibit any attempt to identify survey participants.

## Quick checksum check

```bash
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" data/brfss/main_data_brfss_2024.XPT
```
On Windows PowerShell: `Get-FileHash data\brfss\main_data_brfss_2024.XPT -Algorithm SHA256`.
