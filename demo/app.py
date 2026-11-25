"""Streamlit demo application for Document Intelligence."""

import io
import json
import time
from pathlib import Path

import streamlit as st
import requests
from PIL import Image
import pandas as pd

# Page config
st.set_page_config(
    page_title="Document Intelligence Demo",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"


def check_api_health():
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health/ready", timeout=5)
        return response.status_code == 200 and response.json().get("status") == "ready"
    except Exception:
        return False


def upload_document(file_bytes: bytes, filename: str) -> dict:
    """Upload document to API."""
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    response = requests.post(f"{API_BASE_URL}/documents", files=files, timeout=60)
    response.raise_for_status()
    return response.json()


def get_document(document_id: str) -> dict:
    """Get document details."""
    response = requests.get(f"{API_BASE_URL}/documents/{document_id}", timeout=10)
    response.raise_for_status()
    return response.json()


def get_extraction_result(document_id: str) -> dict:
    """Get extraction results."""
    response = requests.get(f"{API_BASE_URL}/documents/{document_id}/extraction", timeout=10)
    response.raise_for_status()
    return response.json()


def list_documents(status: str = None, page: int = 1, page_size: int = 20) -> dict:
    """List documents."""
    params = {"page": page, "page_size": page_size}
    if status:
        params["status"] = status
    response = requests.get(f"{API_BASE_URL}/documents", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def main():
    st.title("📄 Document Intelligence Demo")
    st.markdown("**Automated Invoice & Form Extraction** powered by LayoutLMv3 + PaddleOCR")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_url = st.text_input("API Base URL", value=API_BASE_URL)

        # Health check
        if st.button("🔍 Check API Health"):
            with st.spinner("Checking..."):
                healthy = check_api_health()
                if healthy:
                    st.success("✅ API is healthy")
                else:
                    st.error("❌ API is not reachable")

        st.divider()

        st.header("📊 Statistics")
        try:
            stats = list_documents(page_size=1)
            st.metric("Total Documents", stats.get("total", 0))
        except Exception:
            st.metric("Total Documents", "N/A")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "📋 Documents", "🔍 Extraction Results", "📈 Analytics"])

    with tab1:
        st.header("Upload Document")
        st.markdown("Supported formats: PDF, PNG, JPG, TIFF (max 10MB)")

        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
            help="Upload an invoice or form document for extraction",
        )

        if uploaded_file is not None:
            # Show preview
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Preview")
                if uploaded_file.type.startswith("image"):
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_column_width=True)
                else:
                    st.info(f"📄 {uploaded_file.name} ({uploaded_file.size:,} bytes)")

            with col2:
                st.subheader("File Info")
                st.json({
                    "filename": uploaded_file.name,
                    "size": f"{uploaded_file.size:,} bytes",
                    "type": uploaded_file.type,
                })

                if st.button("🚀 Process Document", type="primary", use_container_width=True):
                    with st.spinner("Processing document... This may take a few seconds."):
                        try:
                            file_bytes = uploaded_file.getvalue()
                            result = upload_document(file_bytes, uploaded_file.name)
                            st.success(f"✅ Document queued for processing! ID: {result['document_id']}")
                            st.session_state.last_document_id = result["document_id"]
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Upload failed: {str(e)}")

    with tab2:
        st.header("Document List")

        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Filter by Status", ["All", "queued", "processing", "completed", "failed"])
        with col2:
            page_size = st.selectbox("Page Size", [10, 20, 50, 100], index=1)
        with col3:
            page = st.number_input("Page", min_value=1, value=1)

        try:
            status = None if status_filter == "All" else status_filter
            docs = list_documents(status=status, page=page, page_size=page_size)

            if docs["documents"]:
                df = pd.DataFrame(docs["documents"])
                # Format for display
                display_df = df[[
                    "id", "filename", "status", "vendor_name", "invoice_number",
                    "invoice_date", "total_amount", "currency", "created_at"
                ]].copy()

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "id": st.column_config.TextColumn("ID", width="small"),
                        "filename": st.column_config.TextColumn("Filename", width="medium"),
                        "status": st.column_config.SelectboxColumn("Status", options=["queued", "processing", "completed", "failed"]),
                        "vendor_name": st.column_config.TextColumn("Vendor", width="medium"),
                        "invoice_number": st.column_config.TextColumn("Invoice #", width="small"),
                        "invoice_date": st.column_config.DateColumn("Date", width="small"),
                        "total_amount": st.column_config.NumberColumn("Total", format="$%.2f"),
                        "currency": st.column_config.TextColumn("Curr.", width="small"),
                        "created_at": st.column_config.DatetimeColumn("Created", width="medium"),
                    }
                )

                # Pagination
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if page > 1:
                        if st.button("← Previous"):
                            st.session_state.page = page - 1
                            st.rerun()
                with col2:
                    st.write(f"Page {page} of {(docs['total'] + page_size - 1) // page_size} (Total: {docs['total']})")
                with col3:
                    if page * page_size < docs["total"]:
                        if st.button("Next →"):
                            st.session_state.page = page + 1
                            st.rerun()
            else:
                st.info("No documents found")
        except Exception as e:
            st.error(f"Failed to load documents: {str(e)}")

    with tab3:
        st.header("Extraction Results")

        document_id = st.text_input("Document ID", value=st.session_state.get("last_document_id", ""))

        if document_id:
            if st.button("🔍 Load Results"):
                try:
                    with st.spinner("Loading extraction results..."):
                        doc = get_document(document_id)
                        extraction = get_extraction_result(document_id)

                    # Document info
                    st.subheader("Document Info")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Status", doc["status"])
                        st.metric("Vendor", doc["vendor_name"] or "N/A")
                    with col2:
                        st.metric("Invoice #", doc["invoice_number"] or "N/A")
                        st.metric("Date", doc["invoice_date"] or "N/A")
                    with col3:
                        st.metric("Total", f"{doc['currency']} {doc['total_amount']:.2f}" if doc["total_amount"] else "N/A")
                        st.metric("Processing Time", f"{doc['processing_time_ms']}ms" if doc["processing_time_ms"] else "N/A")

                    # Extracted fields
                    st.subheader("Extracted Fields")
                    fields = extraction.get("fields", {})
                    confidence = extraction.get("confidence_scores", {})

                    if fields:
                        field_data = []
                        for key, value in fields.items():
                            field_data.append({
                                "Field": key.replace("_", " ").title(),
                                "Value": value,
                                "Confidence": f"{confidence.get(key, 0):.2%}",
                            })
                        st.dataframe(pd.DataFrame(field_data), use_container_width=True, hide_index=True)
                    else:
                        st.warning("No fields extracted")

                    # Confidence visualization
                    if confidence:
                        st.subheader("Field Confidence Scores")
                        conf_df = pd.DataFrame(list(confidence.items()), columns=["Field", "Confidence"])
                        conf_df["Field"] = conf_df["Field"].str.replace("_", " ").str.title()
                        st.bar_chart(conf_df.set_index("Field"))

                    # Raw OCR text (expandable)
                    with st.expander("📝 Raw OCR Text"):
                        st.text_area("", extraction.get("raw_ocr_text", ""), height=200)

                    # Raw entities (expandable)
                    with st.expander("🏷️ Raw Entities"):
                        st.json(extraction.get("raw_entities", []))

                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        st.error("Document not found")
                    elif e.response.status_code == 400:
                        st.error("Document not processed yet")
                    else:
                        st.error(f"Error: {e}")
                except Exception as e:
                    st.error(f"Failed to load results: {str(e)}")

    with tab4:
        st.header("Analytics Dashboard")

        try:
            docs = list_documents(page_size=1000)
            if docs["documents"]:
                df = pd.DataFrame(docs["documents"])

                # Status distribution
                st.subheader("Document Status Distribution")
                status_counts = df["status"].value_counts()
                st.bar_chart(status_counts)

                # Vendor analysis
                if df["vendor_name"].notna().any():
                    st.subheader("Top Vendors")
                    vendor_counts = df["vendor_name"].value_counts().head(10)
                    st.bar_chart(vendor_counts)

                # Processing time
                if df["processing_time_ms"].notna().any():
                    st.subheader("Processing Time Distribution")
                    time_df = df[df["processing_time_ms"].notna()]["processing_time_ms"]
                    st.line_chart(time_df.sort_values())

                # Amount distribution
                if df["total_amount"].notna().any():
                    st.subheader("Invoice Amount Distribution")
                    amount_df = df[df["total_amount"].notna()]["total_amount"]
                    st.line_chart(amount_df.sort_values())
            else:
                st.info("No data available for analytics")
        except Exception as e:
            st.error(f"Failed to load analytics: {str(e)}")


if __name__ == "__main__":
    main()