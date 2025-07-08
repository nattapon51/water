import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.dates as mdates

# -------------------------------
# ดึงข้อมูลจากเว็บไซต์
# -------------------------------
@st.cache_data(ttl=3600)
def fetch_tide_data():
    url = "https://www.thailandtidetables.com/ไทย/ตารางน้ำขึ้นน้ำลง-ปากน้ำบางปะกง-ฉะเชิงเทรา-480.php"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return pd.DataFrame(), f"❌ ดึงข้อมูลไม่ได้: {e}"

    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", {"class": "tide-table"})

    date_text = soup.find("h2") or soup.find("caption") or soup.find("strong")
    if not date_text:
        return pd.DataFrame(), "❌ ไม่พบวันที่จากเว็บไซต์"

    text = date_text.get_text()
    match = re.search(r"วันที่\s*(\d{1,2})\s*(\S+)\s*(\d{4})", text)
    if not match:
        return pd.DataFrame(), "❌ ไม่สามารถอ่านวันที่ได้"

    day, month_th, year = match.groups()
    month_map = {
        "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4,
        "พฤษภาคม": 5, "มิถุนายน": 6, "กรกฎาคม": 7, "สิงหาคม": 8,
        "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12
    }
    month = month_map.get(month_th)
    if not month:
        return pd.DataFrame(), f"❌ ไม่รู้จักเดือน: {month_th}"

    try:
        base_date = datetime(int(year), month, int(day))
    except:
        return pd.DataFrame(), "❌ วันที่ไม่ถูกต้อง"

    rows = table.find_all("tr")
    data = []
    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) >= 2:
            time_str = cols[0].text.strip()
            level_str = cols[1].text.strip().replace("m", "").replace("เมตร", "")
            try:
                dt = datetime.strptime(time_str, "%H:%M")
                full_dt = datetime.combine(base_date, dt.time())
                level = float(level_str)
                data.append({"ds": full_dt, "y": level})
            except:
                continue

    if not data:
        return pd.DataFrame(), "⚠️ ไม่พบข้อมูลที่แปลงได้"

    df = pd.DataFrame(data)
    return df, None

# -------------------------------
# เริ่มต้นแอป Streamlit
# -------------------------------
st.set_page_config(page_title="พยากรณ์น้ำขึ้นน้ำลง", page_icon="🌊")
st.title("🌊 พยากรณ์น้ำขึ้นน้ำลง (รวมข้อมูลจากเว็บ + ไฟล์ที่อัปโหลด)")

# โหลดจากเว็บไซต์
with st.spinner("🌐 โหลดข้อมูลจากเว็บไซต์..."):
    df_web, error = fetch_tide_data()

# อัปโหลดไฟล์
uploaded_files = st.file_uploader("📂 อัปโหลดไฟล์ .tsv (หลายไฟล์ได้)", type="tsv", accept_multiple_files=True)

df_files = []
if uploaded_files:
    for file in uploaded_files:
        try:
            df_temp = pd.read_csv(file, sep="\t")
            df_temp['ds'] = pd.to_datetime(df_temp['ds'], errors='coerce')
            df_temp['y'] = pd.to_numeric(df_temp['y'], errors='coerce')
            df_temp.dropna(inplace=True)
            df_files.append(df_temp)
            st.success(f"✅ โหลดไฟล์: {file.name}")
        except Exception as e:
            st.error(f"❌ โหลดไม่ได้: {file.name} | {e}")

df_file_all = pd.concat(df_files, ignore_index=True) if df_files else pd.DataFrame()

# รวมข้อมูลทั้งหมด
if error:
    st.warning(error)

df_combined = pd.concat([df_web, df_file_all], ignore_index=True).dropna()
df_combined.sort_values("ds", inplace=True)

if df_combined.empty:
    st.warning("⚠️ ไม่มีข้อมูลเพียงพอจากเว็บหรือไฟล์")
else:
    st.subheader("📋 ข้อมูลรวมทั้งหมด")
    st.dataframe(df_combined)

    st.subheader("🔮 พยากรณ์น้ำขึ้นน้ำลง")
    periods = st.slider("พยากรณ์ล่วงหน้ากี่ชั่วโมง?", 6, 72, 24)

    df_past = df_combined[df_combined['ds'] <= datetime.now()]
    if len(df_past) < 10:
        st.warning("⚠️ ข้อมูลย้อนหลังไม่เพียงพอ")
    else:
        model = Prophet()
        with st.spinner("📈 กำลังวิเคราะห์..."):
            model.fit(df_past)
            future = model.make_future_dataframe(periods=periods, freq='H')
            forecast = model.predict(future)

        st.subheader("📈 กราฟการพยากรณ์")

        # --- ค่าระดับ ---
        mean_level = 2.82
        high_level = 3.51
        low_level = 1.90

        # กราฟ
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(forecast['ds'], forecast['yhat'], color='skyblue', label="แนวโน้ม")
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)

        # จุดเตือน
        high_points = forecast[forecast['yhat'] >= high_level]
        low_points = forecast[forecast['yhat'] <= low_level]
        ax.scatter(high_points['ds'], high_points['yhat'], color='red', label="🌊 น้ำขึ้นสูง")
        ax.scatter(low_points['ds'], low_points['yhat'], color='orange', label="⬇️ น้ำลงต่ำ")

        # เส้นระดับ
        ax.axhline(mean_level, color='green', linestyle='--', label="ระดับปกติ (2.82 ม.)")
        ax.axhline(high_level, color='red', linestyle='--', label="⚠️ ≥ 3.51 ม.")
        ax.axhline(low_level, color='orange', linestyle='--', label="⚠️ ≤ 1.90 ม.")

        # แกนและตกแต่ง
        ax.set_title("📈 การพยากรณ์ระดับน้ำ")
        ax.set_ylabel("ระดับน้ำ (เมตร)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
        ax.tick_params(axis='x', rotation=30)
        ax.legend()
        st.pyplot(fig)

        st.subheader("📊 ตารางผลการพยากรณ์ล่าสุด")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # แจ้งเตือน
        max_level = forecast['yhat'].max()
        min_level = forecast['yhat'].min()

        if max_level >= high_level:
            st.error(f"🌊 แจ้งเตือน: น้ำจะขึ้นสูงสุดที่ {max_level:.2f} เมตร")
        if min_level <= low_level:
            st.warning(f"⬇️ แจ้งเตือน: น้ำจะลดต่ำสุดที่ {min_level:.2f} เมตร")
        if low_level < min_level < high_level and max_level < high_level:
            st.success("✅ ระดับน้ำคาดว่าอยู่ในช่วงปกติ")
