import streamlit as st
import pandas as pd
import numpy as np
import pulp
import pydeck as pdk
import graphviz

# Page Styling
st.set_page_config(page_title="SkinIO - C2C Care Matchmaker", layout="wide", page_icon="🩺")

st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    .card { background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 15px; }
    .badge-primary { background-color: #0284c7; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Connection to Care (C2C) Intelligent Matchmaker")
st.caption("Automated Patient-to-Clinic Assignment Engine, Network Factor Graphs & Geolocation Analytics")

if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False

# Sidebar Controls
st.sidebar.header("⚙️ System Control Panel")
num_patients = st.sidebar.slider("Urgent Patient Pool", 10, 150, 40)
num_doctors = st.sidebar.slider("Clinic Network", 5, 25, 12)
seed = st.sidebar.number_input("Random Seed", value=42)

# Manual Intake Modal
with st.sidebar.expander("➕ Add Live Urgent Patient"):
    new_id = f"P_{num_patients + 1:03d}"
    new_ins = st.selectbox("Insurance Network", ['Aetna', 'Cigna', 'BlueCross', 'UnitedHealthcare', 'Medicare'])
    add_btn = st.button("Submit Patient Intake")

if st.sidebar.button("🚀 Run Matchmaker Engine", type="primary") or add_btn:
    np.random.seed(seed)
    insurances = ['Aetna', 'Cigna', 'BlueCross', 'UnitedHealthcare', 'Medicare']

    patients = pd.DataFrame({
        'Patient_ID': [f"P_{i:03d}" for i in range(num_patients)],
        'lat': np.random.uniform(40.65, 40.85, num_patients),
        'lon': np.random.uniform(-74.05, -73.80, num_patients),
        'Insurance': np.random.choice(insurances, num_patients)
    })

    doctors = pd.DataFrame({
        'Clinic_ID': [f"Clinic_{j:02d}" for j in range(num_doctors)],
        'lat': np.random.uniform(40.65, 40.85, num_doctors),
        'lon': np.random.uniform(-74.05, -73.80, num_doctors),
        'Accepted_Insurances': [
            list(np.random.choice(insurances, np.random.randint(2, 5), replace=False)) 
            for _ in range(num_doctors)
        ],
        'Capacity': np.random.randint(4, 7, num_doctors)
    })

    # Optimization Problem
    prob = pulp.LpProblem("C2C_Matching", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("match", ((i, j) for i in patients.index for j in doctors.index), cat='Binary')

    objective_terms = []
    for i in patients.index:
        for j in doctors.index:
            dist = np.sqrt((patients.loc[i, 'lat'] - doctors.loc[j, 'lat'])**2 + 
                           (patients.loc[i, 'lon'] - doctors.loc[j, 'lon'])**2) * 69
            if patients.loc[i, 'Insurance'] in doctors.loc[j, 'Accepted_Insurances']:
                objective_terms.append(dist * x[i, j])
            else:
                prob += x[i, j] == 0

    prob += pulp.lpSum(objective_terms)

    for i in patients.index:
        prob += pulp.lpSum(x[i, j] for j in doctors.index) == 1

    for j in doctors.index:
        prob += pulp.lpSum(x[i, j] for i in patients.index) <= doctors.loc[j, 'Capacity']

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] == 'Optimal':
        primary_matches = {}
        clinic_loads = {j: 0 for j in doctors.index}
        
        for i in patients.index:
            for j in doctors.index:
                if x[i, j].varValue == 1.0:
                    primary_matches[i] = j
                    clinic_loads[j] += 1

        patient_profiles = []
        network_arcs = []

        for i in patients.index:
            p_rec = patients.loc[i]
            compat_clinics = []
            
            for j in doctors.index:
                d_rec = doctors.loc[j]
                if p_rec['Insurance'] in d_rec['Accepted_Insurances']:
                    dist = np.sqrt((p_rec['lat'] - d_rec['lat'])**2 + (p_rec['lon'] - d_rec['lon'])**2) * 69
                    compat_clinics.append({
                        'Clinic_ID': d_rec['Clinic_ID'],
                        'Clinic_Lat': d_rec['lat'],
                        'Clinic_Lon': d_rec['lon'],
                        'Distance': round(dist, 2),
                        'Is_Primary': (j == primary_matches[i])
                    })
            
            compat_clinics = sorted(compat_clinics, key=lambda c: c['Distance'])
            primary = next((c for c in compat_clinics if c['Is_Primary']), compat_clinics[0])
            alternatives = [c for c in compat_clinics if not c['Is_Primary']][:3]
            
            network_arcs.append({
                'from_lat': p_rec['lat'], 'from_lon': p_rec['lon'],
                'to_lat': primary['Clinic_Lat'], 'to_lon': primary['Clinic_Lon'],
                'color': [16, 185, 129, 220]
            })

            patient_profiles.append({
                'Patient_ID': p_rec['Patient_ID'],
                'Insurance': p_rec['Insurance'],
                'lat': p_rec['lat'], 'lon': p_rec['lon'],
                'Primary_Clinic': primary['Clinic_ID'],
                'Primary_Distance': primary['Distance'],
                'Alternatives': alternatives
            })

        doctors['Assigned_Patients'] = [clinic_loads[j] for j in doctors.index]
        st.session_state.patients_df = pd.DataFrame(patient_profiles)
        st.session_state.doctors_df = doctors
        st.session_state.arcs_df = pd.DataFrame(network_arcs)
        st.session_state.data_generated = True

# Main Dashboard View
if st.session_state.data_generated:
    patients_df = st.session_state.patients_df
    doctors_df = st.session_state.doctors_df
    arcs_df = st.session_state.arcs_df

    # Analytics Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Urgent Patients Matched", len(patients_df))
    m2.metric("Available Network Clinics", len(doctors_df))
    m3.metric("Avg Travel Distance", f"{patients_df['Primary_Distance'].mean():.1f} mi")
    m4.metric("Avg Clinic Capacity Load", f"{(doctors_df['Assigned_Patients'].sum() / doctors_df['Capacity'].sum()) * 100:.1f}%")

    st.markdown("---")

    # Layout Tabs
    tab_inspect, tab_topology, tab_map, tab_analytics = st.tabs([
        "🔍 Patient Inspector", 
        "🕸️ Factor Network Graph", 
        "🗺️ Geographic Arc Map", 
        "📊 Network Analytics"
    ])

    with tab_inspect:
        c1, c2 = st.columns([1, 1])
        with c1:
            selected_id = st.selectbox("Select Patient to Review:", patients_df['Patient_ID'])
            row = patients_df[patients_df['Patient_ID'] == selected_id].iloc[0]

            st.markdown(f"""
            <div class="card">
                <h3>Patient Dossier: {row['Patient_ID']}</h3>
                <p><strong>Insurance Coverage:</strong> {row['Insurance']}</p>
                <p><strong>Primary Choice:</strong> <span class="badge-primary">{row['Primary_Clinic']}</span></p>
                <p><strong>Travel Distance:</strong> {row['Primary_Distance']} miles</p>
            </div>
            """, unsafe_allow_html=True)

            st.write("**Alternative Clinic Options (Ordered by Distance & Insurance Match):**")
            for idx, alt in enumerate(row['Alternatives']):
                st.write(f"• **Option {idx+2}:** {alt['Clinic_ID']} — {alt['Distance']} miles away")

        with c2:
            st.subheader("Manual Supervisor Re-assignment")
            override = st.selectbox("Select Alternative Clinic Override:", [row['Primary_Clinic']] + [a['Clinic_ID'] for a in row['Alternatives']])
            if st.button("Confirm Override"):
                st.success(f"Re-assigned {row['Patient_ID']} to {override}")

    with tab_topology:
        st.subheader("🕸️ Factor & Constraint Routing Topology")
        st.caption("Visualizes the non-geographic decision network connecting Patient Profiles, Constraints (Insurance Match, Capacity), and Clinics.")

        # Construct Graphviz Dependency Network
        graph = graphviz.Digraph(engine="dot")
        graph.attr(bgcolor="#0f172a", rankdir="LR")
        graph.attr('node', shape='box', style='filled', fontname='Helvetica', fontcolor='white')

        # Add Nodes & Flow logic for selected patient
        sel_row = patients_df[patients_df['Patient_ID'] == selected_id].iloc[0]
        p_node = f"Patient: {sel_row['Patient_ID']}"
        ins_node = f"Insurance: {sel_row['Insurance']}"
        prim_node = f"Primary: {sel_row['Primary_Clinic']}"

        graph.node(p_node, fillcolor="#ef4444")
        graph.node(ins_node, fillcolor="#3b82f6")
        graph.node(prim_node, fillcolor="#10b981")

        graph.edge(p_node, ins_node, label="Filter Compatibility")
        graph.edge(ins_node, prim_node, label="Min Distance Solution")

        for alt in sel_row['Alternatives']:
            alt_node = f"Backup: {alt['Clinic_ID']}"
            graph.node(alt_node, fillcolor="#64748b")
            graph.edge(ins_node, alt_node, label=f"{alt['Distance']} mi")

        st.graphviz_chart(graph)

    with tab_map:
        st.subheader("🗺️ Geographic Spatial Distribution Map")
        view_state = pdk.ViewState(
            latitude=patients_df['lat'].mean(),
            longitude=patients_df['lon'].mean(),
            zoom=10, pitch=40
        )
        arc_layer = pdk.Layer("ArcLayer", data=arcs_df, get_source_position=["from_lon", "from_lat"], get_target_position=["to_lon", "to_lat"], get_color="color", get_width=2)
        patient_layer = pdk.Layer("ScatterplotLayer", data=patients_df, get_position=["lon", "lat"], get_color="[239, 68, 68, 200]", get_radius=300)
        clinic_layer = pdk.Layer("ScatterplotLayer", data=doctors_df, get_position=["lon", "lat"], get_color="[16, 185, 129, 250]", get_radius=600)
        st.pydeck_chart(pdk.Deck(layers=[arc_layer, patient_layer, clinic_layer], initial_view_state=view_state))

    with tab_analytics:
        st.subheader("📊 Clinic Capacity Utilization")
        st.bar_chart(doctors_df.set_index('Clinic_ID')[['Capacity', 'Assigned_Patients']])
        
        # Download Dispatch Data
        csv_data = patients_df[['Patient_ID', 'Insurance', 'Primary_Clinic', 'Primary_Distance']].to_csv(index=False)
        st.download_button("📥 Export Final Dispatch Schedule (CSV)", data=csv_data, file_name="c2c_assignments.csv", mime="text/csv")
