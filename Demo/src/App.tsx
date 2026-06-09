/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import {
  UploadCloud,
  ScanSearch,
  Crop,
  Wand2,
  Type,
  Image as ImageIcon,
  ChevronDown,
  Download,
  CheckCircle,
  AlertTriangle,
  Copy,
  Edit2,
  Code2,
  FileText
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { Client } from "@gradio/client";

const GRADIO_URL = import.meta.env.VITE_GRADIO_URL || "https://580b460de234765ae1.gradio.live";

export default function App() {
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState('SVTR');

  const [uploadedImagePreview, setUploadedImagePreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // State để lưu kết quả trả về từ Kaggle
  const [results, setResults] = useState<any>(null);

  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const models = [
    { id: 'CRNN', name: 'CRNN OCR' },
    { id: 'SVTR', name: 'SVTR OCR' },
    { id: 'TrOCR', name: 'trOCR (Transformer)' }
  ];

  // Chỉ giữ 1 hàm handleFileChange duy nhất này
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file); // Lưu lại file gốc để đẩy lên server
      setUploadedImagePreview(URL.createObjectURL(file)); // Preview UI
    }
  };

  // Hàm gọi Gradio API
  const handleProcessImage = async () => {
    if (!selectedFile) {
      alert("Vui lòng tải ảnh lên trước!");
      return;
    }

    setIsProcessing(true);
    try {
      // Thay URL này bằng public link bạn nhận được từ Kaggle
      const app = await Client.connect(GRADIO_URL);

      const result = await app.predict("/process_image", [
        selectedFile,      // inp_image
        selectedModel      // inp_model
      ]);

      console.log("Gradio Output:", result.data);
      // result.data[0]: Ảnh đã vẽ bounding box
      // result.data[1]: Gallery các ảnh crop và text OCR
      // result.data[2]: Chuỗi string log

      setResults(result.data);

    } catch (error) {
      console.error("Lỗi khi kết nối Gradio:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar - LUMINA STYLE */}
      <nav className="fixed left-0 top-0 h-full w-64 border-r border-outline-variant bg-surface flex flex-col z-50">
        <div className="p-6 mb-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <div className="w-4 h-4 bg-white rounded-sm"></div>
            </div>
            <span className="text-xl font-bold tracking-tight text-on-surface">MangaOCR</span>
          </div>
          <div className="text-on-surface-variant text-[10px] uppercase font-bold tracking-widest mt-4 pl-1">Processing Pipeline</div>
        </div>

        <div className="flex flex-col gap-1 font-sans text-sm antialiased flex-grow">
          <div className="px-6 mb-6">
            <label className="block text-[10px] font-bold text-outline uppercase tracking-widest mb-2" htmlFor="model-select">
              Model Selection
            </label>
            <div className="relative">
              <button
                onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
                className="w-full bg-white border border-outline-variant rounded-lg px-4 py-2 text-[11px] font-bold text-on-surface flex items-center justify-between hover:bg-slate-50 transition-all cursor-pointer focus:ring-2 focus:ring-primary/20 shadow-sm"
              >
                <span className="truncate">{models.find(m => m.id === selectedModel)?.name}</span>
                <ChevronDown size={12} className={`transition-transform duration-200 text-outline ${isModelDropdownOpen ? 'rotate-180' : ''}`} />
              </button>

              <AnimatePresence>
                {isModelDropdownOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -4, scale: 0.95 }}
                    animate={{ opacity: 1, y: 4, scale: 1 }}
                    exit={{ opacity: 0, y: -4, scale: 0.95 }}
                    className="absolute top-full left-0 right-0 bg-white border border-outline-variant rounded-xl shadow-[0_10px_25px_-5px_rgba(0,0,0,0.1),0_8px_10px_-6px_rgba(0,0,0,0.1)] z-[60] overflow-hidden py-1.5 mt-1 border-t-0"
                  >
                    {models.map(model => (
                      <button
                        key={model.id}
                        onClick={() => {
                          setSelectedModel(model.id);
                          setIsModelDropdownOpen(false);
                        }}
                        className={`w-full text-left px-4 py-2 text-[11px] font-bold transition-all flex items-center justify-between ${selectedModel === model.id
                          ? 'bg-indigo-50 text-indigo-600 border-r-2 border-primary'
                          : 'text-on-surface-variant hover:bg-slate-50 hover:text-on-surface'
                          }`}
                      >
                        {model.name}
                        {selectedModel === model.id && <CheckCircle size={10} className="text-primary" />}
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          <div className="flex-1">
            <div className="sidebar-active flex items-center px-6 py-3 cursor-pointer">
              <ImageIcon size={16} className="mr-3" />
              <span className="text-sm font-semibold">Active Session</span>
            </div>
            <div className="flex items-center px-6 py-3 text-on-surface-variant hover:bg-surface-container-low cursor-pointer transition-colors">
              <FileText size={16} className="mr-3" />
              <span className="text-sm font-medium">Batch History</span>
            </div>
          </div>

          {/* System Status - LUMINA STYLE */}
          <div className="p-6 border-t border-outline-variant/30">
            <div className="bg-surface-container-low rounded-xl p-4 border border-outline-variant/50">
              <div className="text-[10px] uppercase tracking-wider font-bold text-outline mb-2">Engine Status</div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-success"></div>
                <span className="text-xs font-medium text-on-surface">System Online</span>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="ml-64 w-[calc(100%-16rem)] flex flex-col h-full bg-background relative">
        {/* Header */}
        <header className="fixed top-0 right-0 w-[calc(100%-16rem)] h-16 bg-white border-b border-outline-variant z-40 flex items-center justify-between px-8 shadow-[var(--shadow-header)]">
          <h1 className="text-lg font-semibold text-on-surface">MangaOCR Analysis Dashboard</h1>
          <div className="flex items-center gap-4">
            <div className="flex gap-1">
              <button className="px-3 py-1.5 text-xs font-medium border border-outline-variant rounded-md bg-white hover:bg-surface-container-low text-on-surface">Last 24 Hours</button>
            </div>
            <div className="flex items-center gap-2 ml-4">
              <div className="text-[10px] font-bold text-outline uppercase tracking-widest bg-slate-50 px-2 py-1 rounded border border-outline-variant/30">
                PRO-ENVIRONMENT
              </div>
            </div>
          </div>
        </header>

        {/* Scrollable Area */}
        <main className="mt-16 w-full h-[calc(100vh-4rem)] overflow-y-auto bg-[#F8FAFC]">
          <div className="max-w-[1000px] mx-auto p-8 space-y-8 pb-32">

            {/* Pipeline Overview Diagram */}
            <section className="space-y-4">
              <h2 className="text-sm font-bold text-on-surface uppercase tracking-wide">Analysis Pipeline Status</h2>
              <div className="card-polish p-6 flex items-center justify-between relative overflow-hidden bg-white">
                <div className="absolute top-1/2 left-12 right-12 h-[1px] bg-outline-variant -translate-y-[100%] z-0"></div>

                <PipelineStep active icon={<UploadCloud size={16} />} label="Upload" />
                <PipelineStep icon={<ScanSearch size={16} />} label="Detect" />
                <PipelineStep icon={<Crop size={16} />} label="Crop" />
                <PipelineStep icon={<Type size={16} />} label="OCR" />
              </div>
            </section>

            {/* 1. Image Upload Section */}
            <section className="card-polish overflow-hidden">
              <div className="border-b border-outline-variant px-6 py-4 bg-surface flex justify-between items-center">
                <h3 className="text-sm font-bold text-on-surface uppercase tracking-wide flex items-center gap-3">
                  <UploadCloud className="text-primary" size={18} />
                  1. Content Ingestion
                </h3>
                {/* Bạn có thể thêm onClick={() => { setSelectedFile(null); setUploadedImagePreview(null); }} vào nút Reset này nếu muốn chức năng xóa ảnh */}
                <button className="text-xs text-primary font-semibold hover:underline">Reset Source</button>
              </div>
              <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6 bg-white">
                <div className="border-2 border-dashed border-outline-variant rounded-2xl flex flex-col items-center justify-center p-10 bg-slate-50/50 hover:bg-primary-container/20 transition-all text-center group h-[300px]">
                  <div className="w-12 h-12 rounded-xl bg-white border border-outline-variant shadow-sm flex items-center justify-center mb-4 group-hover:scale-110 transition-transform text-primary">
                    <UploadCloud size={24} />
                  </div>
                  <p className="text-base font-bold text-on-surface">Ingest Raw Page Data</p>
                  <p className="text-xs text-on-surface-variant mt-1">Accepts JPG, PNG up to 10MB</p>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept="image/*"
                    className="hidden"
                  />

                  {/* Cụm nút bấm được cập nhật */}
                  <div className="flex gap-4 mt-6">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="px-6 py-2.5 bg-white border border-outline-variant text-on-surface rounded-xl text-xs font-bold hover:bg-slate-50 transition-colors shadow-sm"
                    >
                      Select Assets
                    </button>

                    <button
                      onClick={handleProcessImage}
                      disabled={isProcessing || !selectedFile}
                      className="px-6 py-2.5 bg-primary text-white rounded-xl text-xs font-bold hover:bg-indigo-600 transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isProcessing ? "Đang xử lý (GPU)..." : "Chạy AI Pipeline"}
                    </button>
                  </div>

                </div>
                <div className="bg-slate-50 border border-outline-variant rounded-2xl p-4 flex items-center justify-center h-[300px] overflow-hidden">
                  {uploadedImagePreview ? (
                    <img
                      src={uploadedImagePreview}
                      alt="Manga Preview"
                      className="max-h-full w-auto object-contain rounded-lg shadow-sm mix-blend-multiply"
                    />
                  ) : (
                    <div className="text-center text-on-surface-variant">
                      <p>Upload a manga page</p>
                      <p className="text-xs mt-1">to preview here</p>
                    </div>
                  )}
                </div>
              </div>
            </section>

            {/* 2. Text Detection Section */}
            {/* 2. Text Detection Section */}
            <section className="card-polish overflow-hidden">
              <div className="border-b border-outline-variant px-6 py-4 bg-surface flex justify-between items-center">
                <h3 className="text-sm font-bold text-on-surface uppercase tracking-wide flex items-center gap-3">
                  <ScanSearch className="text-primary" size={18} />
                  2. Neural Geometry Mapping
                </h3>
              </div>
              <div className="p-8 flex justify-center bg-slate-50/50 min-h-[300px]">
                <div className="relative inline-block border border-outline-variant rounded-2xl p-4 bg-white shadow-xl">
                  {/* Trích xuất ảnh vẽ Bounding Box từ results[0] */}
                  {results && results[0] ? (
                    <img
                      src={results[0]?.url || results[0]}
                      alt="Detection Preview"
                      className="max-h-[400px] w-auto object-contain rounded-lg mix-blend-multiply"
                    />
                  ) : (
                    <div className="flex items-center justify-center h-[300px] w-[300px] md:w-[500px] border-2 border-dashed border-outline-variant/50 rounded-xl text-on-surface-variant text-sm font-medium">
                      {isProcessing ? "Đang nhận diện khung chứa chữ..." : "Chưa có dữ liệu. Vui lòng chạy Pipeline."}
                    </div>
                  )}
                </div>
              </div>
            </section>

            {/* 3. Snippet Cropping Section */}
            <section className="card-polish overflow-hidden">
              <div className="border-b border-outline-variant px-6 py-4 bg-surface flex justify-between items-center">
                <h3 className="text-sm font-bold text-on-surface uppercase tracking-wide flex items-center gap-3">
                  <Crop className="text-primary" size={18} />
                  3. Snippet Isolation
                </h3>
              </div>
              <div className="p-6 bg-white">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
                  {/* Lặp qua mảng gallery crop từ results[1] */}
                  {results && results[1] ? (
                    results[1].map((item: any, index: number) => (
                      <SnippetCard
                        key={index}
                        id={`BLOCK_${String(index + 1).padStart(3, "0")}`}
                        img={(item?.image?.url || item?.image || "") as string}
                      />
                    ))
                  ) : (
                    <div className="col-span-full flex items-center justify-center h-32 bg-slate-50 rounded-2xl border border-dashed border-outline-variant/50 text-on-surface-variant text-sm font-medium">
                      {isProcessing ? "Đang cắt các phân vùng chữ..." : "Chưa có dữ liệu crop."}
                    </div>
                  )}
                </div>
              </div>
            </section>

            {/* 4. OCR Results */}
            <section className="card-polish overflow-hidden">
              <div className="border-b border-outline-variant px-6 py-4 bg-surface flex justify-between items-center">
                <h3 className="text-sm font-bold text-on-surface uppercase tracking-wide flex items-center gap-3">
                  <Type className="text-primary" size={18} />
                  4. Recognition Manifest
                </h3>
                <button className="text-xs text-indigo-600 font-semibold">Validation View</button>
              </div>
              <div className="divide-y divide-outline-variant/30 bg-white">
                {/* Lặp qua mảng gallery text từ results[1] */}
                {results && results[1] ? (
                  results[1].map((item: any, index: number) => {
                    // Gradio trả text trong trường caption
                    const textContent = item?.caption || "Không nhận diện được";
                    return (
                      <ResultRow
                        key={index}
                        id={`BOX_A${index + 1}`}
                        confidence="OK"
                        text={textContent}
                        img={(item?.image?.url || item?.image || "") as string}
                        status="verified"
                      />
                    );
                  })
                ) : (
                  <div className="p-8 text-center text-on-surface-variant text-sm font-medium">
                    {isProcessing ? "Đang xử lý nhận diện ký tự (OCR)..." : "Chưa có kết quả OCR."}
                  </div>
                )}
              </div>
            </section>

            {/* Footer Actions */}
            <div className="flex justify-end items-center gap-4 pt-8 border-t border-outline-variant/30">
              <span className="text-[10px] text-outline font-bold uppercase tracking-widest mr-auto">System v2.1.4 // Build 0423R</span>
              <button className="flex items-center gap-2 px-6 py-2.5 rounded-xl border border-outline-variant bg-white text-on-surface hover:bg-slate-50 transition-all text-xs font-bold shadow-sm active:scale-[0.98]">
                <Code2 size={16} className="text-secondary" />
                Export Raw Data
              </button>
              <button className="flex items-center gap-2 px-8 py-2.5 rounded-xl bg-on-surface text-white hover:bg-slate-800 transition-all text-xs font-bold shadow-md active:scale-[0.98]">
                <Download size={16} />
                Generate Report
              </button>
            </div>
          </div >
        </main >
      </div >
    </div >
  );
}

function PipelineStep({ icon, label, active = false }: { icon: React.ReactNode, label: string, active?: boolean }) {
  return (
    <div className="z-10 flex flex-col items-center gap-3 bg-white px-4">
      <div className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all shadow-sm border ${active ? 'bg-primary text-white border-primary scale-110 shadow-lg' : 'bg-slate-50 text-outline border-outline-variant grayscale opacity-60'}`}>
        {icon}
      </div>
      <span className={`text-[10px] font-bold tracking-widest uppercase ${active ? 'text-primary' : 'text-outline/70'}`}>{label}</span>
    </div>
  );
}

interface SnippetCardProps {
  id: string;
  img: string;
}

function SnippetCard({ id, img }: SnippetCardProps) {
  return (
    <div className="flex flex-col gap-3 group cursor-pointer">
      <div className="bg-slate-50 border border-outline-variant rounded-2xl p-4 flex justify-center items-center h-32 hover:border-primary/50 hover:bg-white hover:shadow-md transition-all overflow-hidden">
        <img
          src={img}
          alt={id}
          className="h-full w-auto object-contain opacity-50 group-hover:opacity-100 group-hover:scale-110 transition-all mix-blend-multiply"
        />
      </div>
      <div className="flex justify-between items-center px-1">
        <span className="font-mono text-[9px] text-on-surface-variant font-bold tracking-tighter uppercase">
          {id}
        </span>
        <button className="text-outline/50 hover:text-primary transition-colors">
          <Download size={14} />
        </button>
      </div>
    </div>
  );
}

interface ResultRowProps {
  id: string;
  confidence: string;
  text: string;
  img: string;
  status: "verified" | "warning";
}

function ResultRow({ id, confidence, text, img, status }: ResultRowProps) {
  return (
    <div className="p-6 flex items-center gap-8 hover:bg-slate-50/50 transition-all group">
      <div className="w-16 h-24 border border-outline-variant/50 rounded-xl bg-white flex items-center justify-center shrink-0 p-2 shadow-sm overflow-hidden group-hover:border-primary/30 transition-colors">
        <img
          src={img}
          alt="Source"
          className="h-full w-auto object-contain opacity-40 group-hover:opacity-100 transition-opacity mix-blend-multiply"
        />
      </div>
      <div className="flex-grow flex flex-col gap-3 min-w-0">
        <div className="flex items-center gap-4">
          <span className="font-mono text-[10px] text-outline font-bold tracking-widest">
            {id}
          </span>
          <span
            className={`px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase tracking-wider flex items-center gap-1 border shadow-sm ${status === "verified"
              ? "bg-green-100 text-green-700 border-green-200"
              : "bg-amber-100 text-amber-700 border-amber-200"
              }`}
          >
            {status === "verified" ? (
              <CheckCircle size={10} />
            ) : (
              <AlertTriangle size={10} />
            )}
            {confidence}
          </span>
        </div>
        <div className="text-base font-medium text-on-surface p-4 bg-slate-50/50 border border-outline-variant/30 rounded-2xl w-full shadow-inner group-hover:border-primary/20 transition-all flex items-center justify-between">
          <span className="truncate">{text}</span>
          <button className="opacity-0 group-hover:opacity-100 p-2 text-primary hover:bg-white rounded-lg shadow-sm transition-all">
            <Edit2 size={14} />
          </button>
        </div>
      </div>
      <button
        className="p-3 text-outline/50 hover:text-primary hover:bg-white rounded-xl shadow-sm transition-all active:scale-90 shrink-0"
        title="Copy Text"
      >
        <Copy size={20} />
      </button>
    </div>
  );
}
