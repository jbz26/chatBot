"use client";

import { useEffect, useRef } from "react";

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  handleSubmit: (e: React.FormEvent) => void;
  handleFileUpload: (file: File) => void;
}

export default function ChatInput({
  input,
  setInput,
  handleSubmit,
  handleFileUpload,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const formRef = useRef<HTMLFormElement | null>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  return (
    <form
      ref={formRef}
      onSubmit={(e) => {
        e.preventDefault(); // Ngăn reload
        handleSubmit(e);    // Gọi hàm xử lý
      }}
      className="p-4"
    >
      <div className="flex items-end bg-[#2a2a2a] px-4 py-2 rounded-2xl w-full max-w-4xl mx-auto shadow-md">
        {/* Nút đính kèm file */}
        <label htmlFor="file-upload" className="cursor-pointer mr-3">
          <div className="w-8 h-8 bg-[#1a1a1a] rounded-md flex items-center justify-center hover:bg-[#333]">
            📎
          </div>
          <input
            id="file-upload"
            type="file"
            accept="*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                handleFileUpload(file);
              }
            }}
          />
        </label>

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          rows={1}
          className="flex-1 bg-transparent text-white outline-none placeholder-gray-400 resize-none max-h-52 overflow-y-auto"
          placeholder="Send a message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              formRef.current?.requestSubmit(); // Gửi form
            }
          }}
        />

        {/* Nút gửi */}
        <button
          type="submit"
          className="ml-2 w-8 h-8 rounded-full bg-gray-500 hover:bg-gray-400 flex items-center justify-center text-white"
        >
          ⬆
        </button>
      </div>
    </form>
  );
}
