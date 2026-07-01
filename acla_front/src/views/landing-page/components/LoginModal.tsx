import React, { useState } from 'react';
import { useAuth } from 'hooks/AuthProvider';

interface LoginModalProps {
  onClose: () => void;
}

const LoginModal: React.FC<LoginModalProps> = ({ onClose }) => {
  const [input, setInput] = useState({ email: '', password: '' });
  const [error, setError] = useState('');
  const auth = useAuth();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (!input.email || !input.password) {
      setError('Please provide email and password.');
      return;
    }
    auth
      .login(input)
      .then(() => onClose())
      .catch(() => setError('Login failed. Please try again.'));
  };

  const handleInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setInput((prev) => ({ ...prev, [name]: value }));
  };

  return (
    <div className="lp-modal__overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="lp-modal__card" onClick={(e) => e.stopPropagation()}>
        <button className="lp-modal__close" onClick={onClose} type="button" aria-label="Close">
          ✕
        </button>
        <h2 className="lp-modal__title">Sign In</h2>
        <form onSubmit={handleSubmit} className="lp-modal__form">
          <label className="lp-modal__label">
            Email
            <input
              className="lp-modal__input"
              type="email"
              name="email"
              placeholder="you@example.com"
              value={input.email}
              onChange={handleInput}
              autoComplete="email"
            />
          </label>
          <label className="lp-modal__label">
            Password
            <input
              className="lp-modal__input"
              type="password"
              name="password"
              placeholder="••••••••"
              value={input.password}
              onChange={handleInput}
              autoComplete="current-password"
            />
          </label>
          {error && <p className="lp-modal__error">{error}</p>}
          <button className="lp-modal__submit" type="submit">
            Sign In
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginModal;
