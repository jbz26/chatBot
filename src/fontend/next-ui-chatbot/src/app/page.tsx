"use client"

import { useState } from 'react'
import ChatMessage from '@/components/ChatMessage'
import ChatInput from '@/components/ChatInput'
import { useEffect } from 'react'
import { useRef } from 'react'

const API_URL = process.env.NEXT_PUBLIC_API_URL!;
console.log("API_URL = ", API_URL)

interface Message {
  role: 'user' | 'assistant';
  text?: string;
  imageUrl?: string;
}

export default function Home() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<Message[]>([
  ])
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    const storedMessages = localStorage.getItem('chatMessages')
    if (storedMessages) {
      setMessages(JSON.parse(storedMessages))
    }
  }, [])

  useEffect(() => {
    localStorage.setItem('chatMessages', JSON.stringify(messages))
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return
    const user_input = input
    setInput('')
    const userMessage: Message = { role: 'user', text: user_input }
    setMessages(prev => [...prev, userMessage])
    
    const res = await fetch(`${API_URL}/user/chat`,
      {
        method:'POST',
        body: JSON.stringify(
          {message: user_input}
        ),headers: {
        "Content-Type": "application/json",
      },
      }
    )
    const data = await res.json();

    const botReply: Message = {
      role: 'assistant',
      text: data.response
    }

    setMessages(prev => [...prev, botReply])

  }

  const handleFileUpload = async (file: File) => {
    const userMessage: Message = { role: 'user', text: 'File uploaded:' + file.name }
    setMessages(prev => [...prev, userMessage])
    const formData = new FormData()
    formData.append('file', file)
    console.log(formData)
    const res = await fetch(`${API_URL}/documents/add_files_directly`, {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    const botReply: Message = {
      role: 'assistant',
      text: `Đã cập nhật thêm ${data.new_doc_count} documents`
    }
    setMessages(prev => [...prev, botReply])

  }

  return (
    <div className="bg-black text-white h-screen flex flex-col items-center ">
      <div className="w-[60%] h-full flex flex-col">
        <div className="flex-1 overflow-y-auto p-4 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
          {messages.map((msg, idx) => (
            <div key={idx}>
              {msg.text && <ChatMessage role={msg.role} text={msg.text} />}
              {msg.imageUrl && (
                <div className={`my-2 ${msg.role === 'user' ? 'text-right' : ''}`}>
                  <img src={msg.imageUrl} alt="uploaded" className="max-w-xs rounded-lg inline-block" />
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>


        <ChatInput
          input={input}
          setInput={setInput}
          handleSubmit={handleSubmit}
          handleFileUpload={handleFileUpload}
        />
      </div>
    </div>
  )
}
