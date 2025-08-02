import { FaRobot } from 'react-icons/fa'; // 👈 Icon robot

interface ChatMessageProps {
  role: 'user' | 'assistant';
  text: string;
}

export default function ChatMessage({ role, text }: ChatMessageProps) {
  const isUser = role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className="flex items-start gap-2">
        {!isUser && (
          <div className="mt-1 text-white">
            <FaRobot size={18} />
          </div>
        )}
        <div
          className={`max-w-xs px-4 py-2 rounded-lg ${
            isUser ? 'bg-white text-black' : 'text-white'
          }`}
        >
          {text}
        </div>
      </div>
    </div>
  );
}
